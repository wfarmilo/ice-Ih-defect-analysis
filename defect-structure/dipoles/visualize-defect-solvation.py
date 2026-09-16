"""
This program takes a given trajectory and trims it down to 
the first and second nearest neighbours of the defect oxygens.

Then, it combines the defect structure positions with a hydrogen
bond network direction for each O, which is the vector sum of all
of the directed hydrogen bonds
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("from_file", help = 'JSON file that points to dataset to read for defect analysis. Output will be saved to data-cache/visualizations')

args = parser.parse_args()

import numpy as np
import MDAnalysis as mda
from icedfmods.Defect_tracking import get_oxyNeighborList
import json, re
from pathlib import Path
from tqdm import tqdm

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
runname = runparams["name"]
dt = runparams["dt"]

# Iterating - only use first system here
dftypes = ["p0m1", "p1m0"]
run_num = runparams["run_indices"][0]
T = runparams["temperature"][0]

# Get input files
pdbin_dir = Path(runparams["pdb_input_dir"])
data_dir = Path(runparams["input_dir"])

# Get output directory
out_dir = Path("../data-cache") / runparams["parent_folder"]

for dft in dftypes:
    # Get pdb input (determined by pXmY-ZZ)
    pdbname = f"{dft}-{run_num:02d}.pdb"
    pdbin = pdbin_dir / pdbname
    ref_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims

    # Make parent directory (pxmY-ZZ)
    out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

    # Find full paths
    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    traj_file = dir_in / f'traj-{dir_in.name}.dcd'
    df_file = out_dir_rich / f'{dir_in.name}-defects.npz'
    HBN_file = out_dir_rich / f'{dir_in.name}-HBN.npz'

    file_out = out_dir_rich / f'{dir_in.name}-visual-dipole.extxyz'
    if file_out.exists(): file_out.unlink()

    # Set up Universe
    u = mda.Universe(pdbin.absolute(), traj_file.absolute(), format = 'dcd')
    u.dimensions = ref_dims
    box = u.dimensions[:3]

    # Get neighbourlist
    oxyNL = get_oxyNeighborList(u)

    # Build translator to Universe indices and back
    oxy = u.select_atoms('name O')
    oxy2uni = oxy.indices
    uni2oxy = np.full(len(u.atoms) + 1, -1, dtype = int)
    uni2oxy[oxy.indices] = np.arange(len(oxy))

    # Load data
    df_dict = np.load(df_file)
    hbn_dict = np.load(HBN_file)
    DFTYPES = df_dict["type_names"]
    hbn_full = {key : hbn_dict[key] for key in hbn_dict.keys() if key != "tis"}
    df_full  = {key : df_dict[key]  for key in df_dict.keys()  if key != "type_names"}

    # Loop over (unique) frames
    frames_unique = np.unique(df_full["frame"])
    for fi in tqdm(frames_unique, desc = f"{dft} frames"):
        u.trajectory[fi]
        apos = u.atoms.positions

        inframe = df_full["frame"] == fi

        df_frame = {key : df_full[key][inframe] for key in df_full.keys()}
        hbn_row = {key : hbn_full[key][fi, :] for key in hbn_full.keys()}

        # Initialize outputs for this frame
        index = []
        hyd_indices = []
        dipole = []
        oh_dipole = []
        labels = []
        prios = []
        hyd_labels = []
        hyd_prios = []

        for ni in range(np.count_nonzero(inframe)):
            df_row = {key : df_frame[key][ni] for key in df_dict.keys() if key != "type_names"}

            name = DFTYPES[df_row["type"]]

            atoms = ["atom1"] if not(name == "L") else ["atom1", "atom2"]
            for atom in atoms:
                # Get neighbours (in OXYGEN indices)
                first_nn = oxyNL[uni2oxy[df_row[atom]], :]
                second_nn = oxyNL[first_nn, :]
                second_nn = second_nn[second_nn != uni2oxy[df_row[atom]]]

                # Switch to universe idxs
                first_nn = oxy2uni[first_nn]
                second_nn = oxy2uni[second_nn]

                # Get attached hydrogen (by donor)
                defect_hyds = hbn_row['hyd'][np.isin(hbn_row['donor'], df_row[atom])]
                first_hyds = hbn_row['hyd'][np.isin(hbn_row["donor"], first_nn)]
                second_hyds = hbn_row['hyd'][np.isin(hbn_row["donor"], second_nn)]

                all_hyds = np.array([*defect_hyds, *first_hyds, *second_hyds], dtype = int)
                hyd_indices.extend(all_hyds)    # Appends each hyd to the list instead of a list of hyds
                hyd_labels.extend(np.repeat([name + '_', name + '_SS1', name + '_SS2'], [defect_hyds.size, first_hyds.size, second_hyds.size]))
                hyd_prios.extend(np.repeat([0, 1, 2], [defect_hyds.size, first_hyds.size, second_hyds.size]))

                # Compute hbn and HOH dipole for each oxygen of interest
                all_oxygens = np.array([df_row[atom], *first_nn, *second_nn], dtype = int)
                shell_labels = np.repeat([name + '_', name + '_SS1', name + '_SS2'], [1, first_nn.size, second_nn.size])
                shell_prios = np.repeat([0, 1, 2], [1, first_nn.size, second_nn.size])

                for oi, oxi in enumerate(all_oxygens):
                    # Find where oxi lies in our hbn
                    isdonor = np.isin(hbn_row["donor"], oxi)
                    isaccep = np.isin(hbn_row["accep"], oxi)

                    # Get position of oxi
                    opos = apos[oxi, :]

                    # Find paired atoms
                    out = hbn_row["accep"][isdonor]
                    ins = hbn_row["donor"][isaccep]

                    # Compute OO donor -> acceptor vectors (hbn dipole)
                    outvecs = apos[out, :] - opos[None, :]
                    insvecs = opos[None, :] - apos[ins, :]

                    outvecs -= box[None, :] * np.rint(outvecs / box[None, :])
                    insvecs -= box[None, :] * np.rint(insvecs / box[None, :])

                    # Save results
                    index.append(oxi)
                    dipole.append(outvecs.sum(axis=0) + insvecs.sum(axis = 0))

                    # Find all NN hydrogens
                    out = hbn_row["hyd"][isdonor]
                    ins = hbn_row["hyd"][isaccep]

                    # Compute OH donor -> acceptor vectors (true unscaled dipole)
                    outvecs = apos[out, :] - opos[None, :]
                    insvecs = opos[None, :] - apos[ins, :]

                    outvecs -= box[None, :] * np.rint(outvecs / box[None, :])
                    insvecs -= box[None, :] * np.rint(insvecs / box[None, :])

                    oh_dipole.append(outvecs.sum(axis=0) + insvecs.sum(axis = 0))

                labels.extend(shell_labels)
                prios.extend(shell_prios)

        # Remove duplicates
        hyd_unique, hyd_unique_inv = np.unique(hyd_indices, return_inverse=True)
        oxy_unique, unique_idxs, unique_inv = np.unique(index, return_index=True, return_inverse = True)
        dip_unique = np.array(dipole)[unique_idxs]
        oh_dip_unique = np.array(oh_dipole)[unique_idxs]

        # Prioritize central labels
        unique_inv = unique_inv.ravel()
        order = np.lexsort((prios, unique_inv))     # Sorts by inverse, then priority (e.g. (4, "OH"), (4, "L SS1"), (4, "L SS2"))
        _, first_occ = np.unique(unique_inv[order], return_index=True)  # Grabs first occurence of each inverse
        best_labels = np.asarray(labels)[order][first_occ]

        hyd_unique_inv = hyd_unique_inv.ravel()
        order = np.lexsort((hyd_prios, hyd_unique_inv))
        _, first_occ = np.unique(hyd_unique_inv[order], return_index=True)
        hyd_best_labels = np.asarray(hyd_labels)[order][first_occ]

        # Convert label to integer IDs
        type_label = {name: i for i, name in enumerate(DFTYPES)}
        shell_label = {'' : 0, 'SS1' : 1, 'SS2' : 2}

        # Get useful extxyz properties
        Na = len(hyd_unique) + len(oxy_unique)
        extxyzheader = f'Lattice=\"{box[0]} 0.0 0.0 0.0 {box[1]} 0.0 0.0 0.0 {box[2]}\" Properties=species:S:1:pos:R:3:OO_dipoles:R:3:OH_dipoles:R:3:color:R:3:defect_type:I:1:shell_id:I:1'
        clrs_default = {
            "OH" : np.array([0, 0, 125])/255, 
            "L" : np.array([0, 125, 0])/255, 
            "H3O" : np.array([125, 0, 0])/255, 
            "D" : np.array([0, 125, 125])/255
        }
        clrs =  {key + '_' : clrs_default[key] for key in clrs_default.keys()} | \
                {key + '_SS1' : np.clip(clrs_default[key] * 1.5, 0, 1) for key in clrs_default.keys()} | \
                {key + '_SS2' : np.clip(clrs_default[key] * 3, 0, 1) for key in clrs_default.keys()}

        # Write this frame to the extxyz
        with open(file_out, "a") as f:
            f.write(f'{Na}\n')
            f.write(f'{extxyzheader}\n')

            # No dipoles for hydrogens
            for hi, li in zip(hyd_unique, hyd_best_labels):
                lis = li.split('_')
                f.write("H " + (12*"{:16.9f} ").format(*apos[hi, :], *[0, 0, 0], *[0, 0, 0], *[1, 1, 1]) + (2*'{:8d}').format(type_label[lis[0]], shell_label[lis[1]]) +'\n')

            # Save dipoles for oxygens
            for oi, di, hdi, li in zip(oxy_unique, dip_unique, oh_dip_unique, best_labels):
                lis = li.split('_')
                f.write("O " + (12*"{:16.9f} ").format(*apos[oi, :], *di, *hdi, *clrs[li]) + (2*'{:8d}').format(type_label[lis[0]], shell_label[lis[1]]) +'\n')

                