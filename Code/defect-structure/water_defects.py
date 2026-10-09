import argparse
from icedfmods.Helper_modules import DATA_CACHE

def_keyfile = DATA_CACHE / "templates/water-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from icedfmods.Defect_tracking import run_waterdefect_tracking
from pathlib import Path
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Received from main: data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all
def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, T, runparams, is_run_all):

    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    traj_file = dir_in / f'traj-{dir_in.name}.dcd'
    df_save = out_dir_rich / f'{dir_in.name}-defects.npz'

    # If in update mode and all output files found, skip it
    if not(is_run_all) and df_save.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # Set up simulation
    u = mda.Universe(pdbin.absolute(), traj_file.absolute(), format = 'dcd')
    u.dimensions = cell_dims
    u.trajectory.add_transformations(trans.wrap(u.atoms, compound = 'atoms'))

    tis = np.arange(len(u.trajectory))

    # Track all defects
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')
    df_dict = run_waterdefect_tracking(u, tis, isverbose = False)

    # Save output
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

    # Use only defect systems
    if "p0m0" in dftypes: dftypes.remove("p0m0")

    # Simulation variables: not the main thing we want to converge, but different cases of it to compare over
    temps = runparams["temperature"]

    # Get input files
    pdbin_dir = Path(runparams["pdb_input_dir"])
    data_dir = Path(runparams["input_dir"])

    pdbin_fmt = runparams["pdb_fmt"]

    # Get output directory
    out_dir = DATA_CACHE / runparams["parent_folder"]
    if not(out_dir.exists()): out_dir.mkdir()

    # Prepare the inputs for each run

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:
            # Get pdb input (determined by pXmY-ZZ)
            pdbname = re.sub(r'XXX[^X]*XXX', '{}', pdbin_fmt).format(dft, f"{run_num:02d}")
            pdbin = pdbin_dir / pdbname
            ref_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims

            # Make parent directory (pxmY-ZZ)
            out_dir_rich = out_dir / f"{dft}-{run_num:02d}"
            if not(out_dir_rich.exists()): out_dir_rich.mkdir()

            # Continue iterating over dependents
            for T in temps:
                # Save inputs for each given task
                task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(temps)])
    max_workers = np.min([tot_workers, 28])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())

if __name__ == '__main__':
    main()