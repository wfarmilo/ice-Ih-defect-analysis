"""
Helper modules to clean up my code
"""

import json
import numpy as np
from pathlib import Path
import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor, as_completed


def submit_parallel_processes(args, cachePath, run_single_file):
    """
    This module takes an argparser input in my standard format and
    unpacks all classical trajectories as indictated in args.from_file's
    JSON key list. Then, it submits up to 28 parallel tasks of the run_single_file
    method which should all be independent
    """

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

    # Get input files
    pdbin_dir = Path(runparams["pdb_input_dir"])
    data_dir = Path(runparams["input_dir"])

    # Get output directory
    out_dir = cachePath / runparams["parent_folder"]
    if not(out_dir.exists()): out_dir.mkdir()

    # Prepare the inputs for each run

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:
            # Get pdb input (determined by pXmY-ZZ)
            pdbname = f"{dft}-{run_num:02d}.pdb"
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