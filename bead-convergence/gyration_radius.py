"""
This program aims to find the radius of gyration between the beads of a ring
polymer and its centroid, for both the averaged entire system and for
defect protons identified from the bead-0 trajectory
"""


import argparse

def_keyfile = "./data-cache/bead-convergence.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)
parser.add_argument("-dfs", "--defects", nargs = "*", help = "Which defects to analyze (forces update mode on them)", default = False)

args = parser.parse_args()

import json
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pathlib import Path
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, pit, T, P, runparams, run_all, run_defects):

    # Name keys for convenience
    st =  '_std'
    o = 'O_'
    h = 'H_'


    # Get home directory from run params
    inputmap = [dft, f"{run_num:02d}", pit, T, P]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    # Input files
    cent_file = dir_in / f'{dir_in.name}.cent.xyz'
    bead_files = [dir_in / f'{dir_in.name}.pos_{i:0{1 + int(P > 9)}d}.extxyz' for i in range(P)]  #2-digit bead counts have 2-digit indices
    dfs_in = out_dir_rich / f'{dir_in.name}-defects.npz'
    hbn_in = out_dir_rich / f'{dir_in.name}-HBN.npz'

    # Output files
    gyr_out = out_dir_rich / f'{dir_in.name}-gyration-radius.npz'

    # If in update mode and any file found, skip it
    if not(run_defects) and not(run_all) and gyr_out.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # If looking for a specific defect update, make sure the file exists
    if run_defects:
        assert gyr_out.exists() and gyr_out.exists(), "Cannot run in defect-specific update mode without an existing file"

    # Get hydrogen bond network
    hbn = np.load(hbn_in)

    # Get mean position (centroid)
    u_cent = mda.Universe(pdbin.absolute(), cent_file.absolute())
    u_cent.dimensions = cell_dims
    u_cent.trajectory.add_transformations(trans.wrap(u_cent.atoms, compound = 'atoms'))

    # For each bead, get its positions
    u_beads = [[] for p in range(P)]
    for p in range(P):
        u_beads[p] = mda.Universe(pdbin.absolute(), bead_files[p].absolute(), format = 'xyz')
        u_beads[p].dimensions = cell_dims
        u_beads[p].trajectory.add_transformations(trans.wrap(u_beads[p].atoms, compound = 'atoms'))

    # Get delta for all defects, iteratively by name
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')

    # Get useful properties
    Na = len(u_cent.atoms)
    Nt = len(u_cent.trajectory)
    box = cell_dims[:3]

    # Get entire system atomwise and timestepwise radius of gyration
    gyr_all = np.zeros((Nt, Na), dtype = float)
    total_beadcount = P

    for ti in range(Nt):
        u_cent.trajectory[ti]
        for p in range(P):

            # If file is truncated skip it and modulate gyr_all accordingly
            if ti >= len(u_beads[p].trajectory):
                # If this is the end of this file, then divide gyr_all by smthg else
                if ti == len(u_beads[p].trajectory):
                    gyr_all[ti, :] *= total_beadcount
                    total_beadcount -= 1
                    gyr_all[ti, :] /= total_beadcount
                continue

            u_beads[p].trajectory[ti]
            dist = u_beads[p].atoms.positions - u_cent.atoms.positions
            dist -= box[None, :] * np.rint(dist[:, :]/box[None, :])

            # Add to total radius of gyration
            gyr_all[ti, :] += np.linalg.norm(dist, axis = -1)**2 / total_beadcount

    # Get bulk averaged radius of gyration for each atom type
    isoxy = u_cent.select_atoms("name O").indices
    ishyd = u_cent.select_atoms("name H").indices

    gyr_avgd = {
        o + 'bulk' : np.mean(gyr_all[:, isoxy]),
        h + 'bulk' : np.mean(gyr_all[:, ishyd]),
        o + 'bulk' + st : np.std(gyr_all[:, isoxy])/np.sqrt(Nt * len(isoxy)),
        h + 'bulk' + st : np.std(gyr_all[:, ishyd])/np.sqrt(Nt * len(ishyd))
    }


    # For each defect, filter out where in gyr_all they appear and average those quantities
    dfs_dict = np.load(dfs_in)

    # Here referring to OH/H3O/L/D
    DFTYPES = dfs_dict['type_names']

    # File saves
    gyr_defect =    {o + name : 0 for name in DFTYPES}          |   \
                    {o + name + st : 0 for name in DFTYPES}     |   \
                    {h + name : 0 for name in DFTYPES}          |   \
                    {h + name + st : 0 for name in DFTYPES}

    # Which defects to track
    dfs_to_modify = run_defects if run_defects else DFTYPES

    # Build a mask of where each defect is in gyr_all
    isdefect = {name : np.zeros((Nt, Na), dtype = bool) for name in DFTYPES}
    isdefect_hyd = {name : np.zeros((Nt, Na), dtype = bool) for name in DFTYPES}

    for name in dfs_to_modify:

        # First, use defect indices directly

        # Get type code from full list
        type_code = list(DFTYPES).index(name)

        # Find atoms of the relevant defect type (OH/H3O/L/D)
        oftype = dfs_dict['type'] == type_code
        frames, atom1s, atom2s = dfs_dict["frame"][oftype], dfs_dict["atom1"][oftype], dfs_dict["atom2"][oftype]

        # Iterate through all found defects and point out where they are in (Nt, Na) space
        isdefect[name][frames, atom1s] = True

        # For L defects, we use both atoms for dfidxs
        if name == "L": isdefect[name][frames, atom2s] = True

        # Then, find bonded hydrogens from HBN
        Ndf = len(frames)
        donors = hbn["donor"][frames, :]        # shape (Ndf, Nh)
        hyds = hbn["hyd"][frames, :]            # shape (Ndf, Nh)
        angles = hbn["HBN_angs"][frames, :]     # shape (Ndf, Nh)

        mask_donor = donors == atom1s[:, None]  # shape (Ndf, Nh)

        match name:
            # Use all for ionic
            case "OH" | "H3O": 
                don_idx, hyd_idx = np.nonzero(mask_donor)
                isdefect_hyd[name][frames[don_idx], hyds[don_idx, hyd_idx]] = True

            # For bjerrum, use most misaligned with its own bond
            case "L":
                # L defect needs to consider both atoms
                mask_donor += donors == atom2s[:, None]

                angles_masked = np.where(mask_donor, angles, np.inf)    # Fill other atoms with infinities
                worst_angle = np.argmin(angles_masked, axis = 1)        # Collapse to (Ndf,)

                isdefect_hyd[name][frames, hyds[np.arange(Ndf), worst_angle]] = True
            case "D":
                angles_masked = np.where(mask_donor, angles, np.inf)    # Fill other atoms with infinities
                worst_angle = np.argmin(angles_masked, axis = 1)        # Collapse to (Ndf,)

                isdefect_hyd[name][frames, hyds[np.arange(Ndf), worst_angle]] = True
        
        assert mask_donor.any(axis=1).all(), f"{dft}-{run_num:02d}/{dir_in.name}: {name} defect not recognized as donor anywhere"

        # Save mean and stderr
        if np.count_nonzero(isdefect[name]) > 0:
            gyr_defect[o + name] = np.mean(gyr_all[isdefect[name]])
            gyr_defect[o + name + st] = np.std(gyr_all[isdefect[name]])/np.sqrt(np.count_nonzero(isdefect[name]))

            gyr_defect[h + name] = np.mean(gyr_all[isdefect_hyd[name]])
            gyr_defect[h + name + st] = np.std(gyr_all[isdefect_hyd[name]])/np.sqrt(np.count_nonzero(isdefect_hyd[name]))
        else:
            # If not found, use -ve placeholder (radius is +ve by construction)
            gyr_defect[o + name] = -1
            gyr_defect[o + name + st] = -1
            gyr_defect[h + name] = -1
            gyr_defect[h + name + st] = -1


    # If we didn't just create all defects, load up the old ones to resave
    if run_defects:
        # Recall that dfs_to_modify is actually the ones we computed
        not_touched = list(set(DFTYPES) - set(dfs_to_modify))

        gyr_cached = np.load(gyr_out)

        # Pull unmodified defects from previously saved values
        for key in not_touched:
            gyr_defect[o + key]         = gyr_cached[o + key] 
            gyr_defect[o + key + st]    = gyr_cached[o + key + st] 
            gyr_defect[h + key]         = gyr_cached[h + key] 
            gyr_defect[h + key + st]    = gyr_cached[h + key + st] 

    # Combine subdicts
    gyr_dict = gyr_avgd | gyr_defect | {"type_names" : DFTYPES}

    # Should only have 12 real values, 6 means + 6 stderrs + dftypes
    np.savez_compressed(gyr_out, **gyr_dict)

    # Result is the diagnostic for printing from the executor
    return f'{dft}-{run_num:02d}/{dir_in.name} completed'

def main():

    # Load run specifications
    run_all = args.all
    run_defects = args.defects

    # Load run info
    with open(args.from_file, "r") as f:
        runparams = json.loads(f.read())

    # Constant across all run types
    name = runparams["name"]
    dt = runparams["dt"]

    # Iterating variables
    dftypes = runparams["defect_types"]     # Here referring to pXmY, the type of defects in my simulaiton
    run_idxs = runparams["run_indices"]

    # Simulation variables: not the main thing we want to converge, but different cases of it to compare over
    temps = runparams["temperature"]
    pitypes = runparams["PIMD_type"]

    # Convergence variable: the dependent variable we are interested in increasing over
    nbeads = runparams["num_beads"]

    # Get input files
    pdbin_dir = Path(runparams["pdb_input_dir"])
    pdb_count = runparams["pdb_atom_count"]
    data_dir = Path(runparams["input_dir"])

    # Get output directory
    out_dir = Path("./data-cache") / runparams["parent_folder"]
    assert out_dir.exists(), f"Directory {out_dir.absolute()} not found. Running from {Path(".").absolute()}"

    # Prepare the inputs for each run

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:

            # Get pdb file name and cell dimensions (determined by pXmY-ZZ)
            pdbname = f"{dft}-{run_num:02d}.pdb" if not(pdb_count) else f"{dft}-{run_num:02d}-N{pdb_count}.pdb"
            pdbin = pdbin_dir / pdbname
            cell_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims
            out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

            # Continue collecting data (which all has the same initial configuration)
            for pit in pitypes:
                for T in temps:
                    for P in nbeads:
                        task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, pit, T, P, runparams, run_all, run_defects))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(pitypes), len(temps), len(nbeads)])
    max_workers = np.min([tot_workers, 28])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())

if __name__ == '__main__':
    main()


