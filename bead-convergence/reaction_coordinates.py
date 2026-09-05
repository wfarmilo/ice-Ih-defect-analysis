import argparse

def_keyfile = "./data-cache/bead-convergence.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)
parser.add_argument("-dfs", "--defects", nargs = "*", help = "Which defects to analyze (forces update mode on them)", default = False)

args = parser.parse_args()

import json
from MDModules import Reaction_coordinates as rc
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pathlib import Path
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, pit, T, P, runparams, run_all, run_defects):

    # Get home directory from run params
    inputmap = [dft, f"{run_num:02d}", pit, T, P]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    # Input files
    bead_file = dir_in / f'{dir_in.name}.pos_0{'' if P < 10 else '0'}.extxyz'  #2-digit bead counts have 2-digit indices
    hbn_in = out_dir_rich / f'{dir_in.name}-HBN.npz'
    dfs_in = out_dir_rich / f'{dir_in.name}-defects.npz'

    # Output files
    rc_delta_out = out_dir_rich / f'{dir_in.name}-delta-coord.npz'
    rc_d_sum_out = out_dir_rich / f'{dir_in.name}-summed-coord.npz'

    # If in update mode and any file found, skip it
    if not(run_defects) and not(run_all) and rc_delta_out.exists() and rc_d_sum_out.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # If looking for a specific defect update, make sure the file exists
    if run_defects:
        assert rc_delta_out.exists() and rc_d_sum_out.exists(), "Cannot run in defect-specific update mode without an existing file"

    # Set up simulation
    u_bead = mda.Universe(pdbin.absolute(), bead_file.absolute(), format = 'xyz')
    u_bead.dimensions = cell_dims
    u_bead.trajectory.add_transformations(trans.wrap(u_bead.atoms, compound = 'atoms'))

    hbn_dict = np.load(hbn_in)
    dfs_dict = np.load(dfs_in)

    tis = hbn_dict["tis"]

    # Get delta for all defects, iteratively by name
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')

    # Here referring to OH/H3O/L/D
    DFTYPES = dfs_dict['type_names']
    deltas = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])
    d_sums = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])

    # Which defects to track
    dfs_to_modify = run_defects if run_defects else dfs_dict['type_names'] 

    delta_method = {'OH' : rc.get_ionic_delta, 'H3O' : rc.get_ionic_delta, 'L' : rc.get_L_delta, 'D' : rc.get_D_delta_atom}
    for name in dfs_to_modify:

        # Get type code from full list
        type_code = list(DFTYPES).index(name)

        # Find atoms of the relevant defect type (OH/H3O/L/D)
        oftype = dfs_dict['type'] == type_code
        frames, atom1s, atom2s = dfs_dict["frame"][oftype], dfs_dict["atom1"][oftype], dfs_dict["atom2"][oftype]

        # For L defects, use both atoms for dfidxs
        zipped = zip(frames, atom1s, atom2s)
        for ts, a1, a2 in zipped:
            u_bead.trajectory[ts]

            # Unshift equilibrium time to find index of hbn
            ti = ts - tis[0]

            # Stack hydrogen bond network into the relevant row (kind of legacy code here)
            hbn_row = np.stack([hbn_dict["hyd"][ti], hbn_dict["donor"][ti], hbn_dict["accep"][ti], hbn_dict["HBN_angs"][ti]])

            # Stack atoms here as well (note that -1 padding will prevent ionic defects from breaking)
            dfs = np.array([a1, a2], dtype = int)

            # Get delta and sum value
            delta, d_sum = delta_method[name](u_bead, dfs, hbn_row)

            # Add to stash (if valid)
            if delta is not None: 
                deltas[name].append(delta)
                d_sums[name].append(d_sum)

    # If we didn't just create all defects, load up the old ones to resave
    if run_defects:
        # Recall that dfs_to_modify is actually the ones we computed
        not_touched = list(set(DFTYPES) - set(dfs_to_modify))

        delta_cached = np.load(rc_delta_out)
        d_sum_cached = np.load(rc_d_sum_out)

        # Pull unmodified defects from previously saved values
        for key in not_touched:
            deltas[key] = delta_cached[key]
            d_sums[key] = d_sum_cached[key]

    np.savez_compressed(rc_delta_out, **deltas)
    np.savez_compressed(rc_d_sum_out, **d_sums)

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


