import argparse

def_keyfile = "./data-cache/bead-convergence.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
from icedfmods.Defect_tracking import run_multidefect_tracking
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pathlib import Path
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, pit, T, P, runparams, is_run_all):

    inputmap = [dft, f"{run_num:02d}", pit, T, P]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    cent_file = dir_in / f'{dir_in.name}.cent.xyz'
    HBN_save = out_dir_rich / f'{dir_in.name}-HBN.npz'
    df_save = out_dir_rich / f'{dir_in.name}-defects.npz'

    # If in update mode and all output files found, skip it
    if not(is_run_all) and (HBN_save.exists() and df_save.exists()):
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # Set up simulation
    u_cent = mda.Universe(pdbin.absolute(), cent_file.absolute(), format = 'xyz')
    u_cent.dimensions = cell_dims
    u_cent.trajectory.add_transformations(trans.wrap(u_cent.atoms, compound = 'atoms'))

    tis = np.arange(len(u_cent.trajectory))

    # Track all defects
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')
    df_dict, hbn_dict = run_multidefect_tracking(u_cent, tis, 20, isverbose = False, recalibrate = True)

    # Save output
    np.savez_compressed(HBN_save, **hbn_dict)
    np.savez_compressed(df_save, **df_dict)

    # Result is the diagnostic for printing from the executor
    return f'{dft}-{run_num:02d}/{dir_in.name} completed'

def main():

    # Load input params
    run_all = args.all

    # Load run info
    with open(args.from_file, "r") as f:
        runparams = json.loads(f.read())

    # Constant across all run types
    name = runparams["name"]
    dt = runparams["dt"]

    # Iterating variables
    dftypes = runparams["defect_types"]
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
    inputdir_raw = runparams["input_dir"]

    # Get output directory
    out_dir = Path("./data-cache") / runparams["parent_folder"]
    if not(out_dir.exists()): out_dir.mkdir()

    # Prepare the inputs for each run

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:
            # Get pdb input (determined by pXmY-ZZ)
            pdbname = f"{dft}-{run_num:02d}.pdb" if not(pdb_count) else f"{dft}-{run_num:02d}-N{pdb_count}.pdb"
            pdbin = pdbin_dir / pdbname
            ref_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims

            # Make parent directory (pxmY-ZZ)
            out_dir_rich = out_dir / f"{dft}-{run_num:02d}"
            if not(out_dir_rich.exists()): out_dir_rich.mkdir()

            # Continue iterating over dependents
            for pit in pitypes:
                for T in temps:
                    for P in nbeads:
                        # Save inputs for each given task
                        task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, pit, T, P, runparams, run_all))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(pitypes), len(temps), len(nbeads)])
    max_workers = np.min([tot_workers, 28])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())

if __name__ == '__main__':
    main()