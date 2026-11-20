import argparse
from icedfmods.Helper_modules import DATA_CACHE


def_keyfile = "water-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
from icedfmods import Cutoff_HBN as cHBN, Reaction_coordinates as rc
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pathlib import Path
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_TASKS = 8

def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, T, runparams, run_all):

    # Get home directory from run params
    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    # Input files
    traj_file = dir_in / f'traj-{dir_in.name}.dcd'
    chbn_in = out_dir_rich / f'{dir_in.name}-cutoff-HBN.npz'

    # Output files
    rc_laage_out = out_dir_rich / f'{dir_in.name}-laage-coord.npy'
    rc_hydid_out = out_dir_rich / f'{dir_in.name}-defect-hydidx.npz'

    # If in update mode and any file found, skip it
    if not(run_all) and rc_laage_out.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # Callout before loading files
    print(f'Reading {dft}-{run_num:02d}/{dir_in.name}')

    # Set up simulation
    u = mda.Universe(pdbin.absolute(), traj_file.absolute(), format = 'dcd')
    u.dimensions = cell_dims
    u.trajectory.add_transformations(trans.wrap(u.atoms, compound = 'atoms'))

    # Callout after dcd not supported spam
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')

    # Minimal loading of big array
    with np.load(chbn_in) as zipped:
        hbonds = {key: zipped[key] for key in ("frame", "donor", "accep", "hyd")}

    # Only exists for H3O
    frames, hydidx = np.load(rc_hydid_out)["H3O"]

    # Get coordinate
    laage_rc, _ = cHBN.get_Laage_coordinate(u, hbonds, hydidx, frames)

    # Save all output as .npz
    np.save(rc_laage_out, np.stack([frames, laage_rc]))

    # Result is the diagnostic for printing from the executor
    return f'{dft}-{run_num:02d}/{dir_in.name} completed'

def main():

    # Load run specifications
    run_all = args.all

    # Load run info
    with open(DATA_CACHE / f"templates/{args.from_file}", "r") as f:
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
    assert out_dir.exists(), f"Directory {out_dir.absolute()} not found. Running from {Path(".").absolute()}"

    # Prepare the inputs for each run

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:

            # Get pdb input (determined by pXmY-ZZ)
            pdbname = re.sub(r'XXX[^X]*XXX', '{}', pdbin_fmt).format(dft, f"{run_num:02d}")
            pdbin = pdbin_dir / pdbname
            ref_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims
            # Read parent directory (pxmY-ZZ)
            out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

            # Continue collecting data (which all has the same initial configuration)
            for T in temps:
                # Save inputs for each task
                task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(temps)])
    max_workers = np.min([tot_workers, MAX_TASKS])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())


    print("Execution complete!")
    #aggregate_files(DATA_CACHE / f"templates/{args.from_file}", out_dir)
    #print(f"Aggregation complete, saving to {out_dir / 'reaction-coord.npz'}")

#TODO: Add Laage data to aggregate?

############### Free energy and file organization ###############################
def aggregate_files(json_path, dir_out):
    coords = rc.load_reaction_coords(json_path)

    # Global properties
    DFTYPES = coords["type_names"]
    temps = coords["unique_temps"]
    run_idxs = np.unique(coords["run_idx"])

    # Initialize saves
    Ndf, Nt, Nri, Nrc = len(DFTYPES), len(temps), len(run_idxs), rc.NBINS - 1
    
    bins = np.full((Ndf, rc.NBINS), np.nan)
    bins2d = np.full((Ndf, Nt, 2, rc.NBINS), np.nan)

    free0s_runs = np.full((Ndf, Nt, Nri), np.nan)
    free1d_runs = np.full((Ndf, Nt, Nri, Nrc), np.nan)
    free2d_runs = np.full((Ndf, Nt, Nrc, Nrc), np.nan)

    # Collect data
    midpoint = rc.NBINS // 2 - 1
    for type_code, name in enumerate(DFTYPES):

        # Mask for defect types
        mask_by_name = coords["type_code"] == type_code

        # Make bins (by defect type)
        maxabs = np.max(np.abs(coords["delta"][mask_by_name]))
        bins[type_code, :] = np.linspace(-maxabs, maxabs, rc.NBINS)

        for ti, T in enumerate(temps):

            # Mask for temperature
            mask_by_T = coords["T"] == T

            for ri, run_idx in enumerate(run_idxs):

                # Mask for run idx
                mask_by_run = coords["run_idx"] == run_idx
                
                # Get reaction coord for this specific (name, T, run_idx)
                delta = coords["delta"][mask_by_name * mask_by_T * mask_by_run]

                # Get free energy
                free1d = rc.get_free_1D(delta, bins[type_code], T)
                free1d_runs[type_code, ti, ri] = free1d

                # Get middle bin value
                free0s_runs[type_code, ti, ri] = free1d[midpoint]


            # Make 2D plot (at all temperatures, but not per run)
            delta = coords["delta"][mask_by_name * mask_by_T]
            omega = coords["omega"][mask_by_name * mask_by_T]

            # Get bins
            maxabs_2d = np.max(np.abs(delta))
            bins_delta = np.linspace(-maxabs_2d, maxabs_2d, rc.NBINS)
            bins_omega = np.linspace(np.min(omega), np.max(omega), rc.NBINS)
            bins2d[type_code, ti, :, :] = np.vstack([bins_delta, bins_omega])

            # Compute free energy
            free2d = rc.get_free_2D(delta, omega, T, bins2d[type_code, ti, :, :])
            free2d_runs[type_code, ti, :, :] = free2d

    # Refactor and save output (axis 2 is the run axis)
    f1_nvalid = np.isfinite(free1d_runs).sum(axis = 2)
    f1_mean = np.nanmean(free1d_runs, axis = 2)
    f1_std = np.nanstd(free1d_runs, axis = 2, ddof = 1) / np.sqrt(f1_nvalid)

    f0_nvalid = np.isfinite(free0s_runs).sum(axis = 2)
    f0_mean = np.nanmean(free0s_runs, axis = 2)
    f0_std = np.nanstd(free0s_runs, axis = 2, ddof = 1) / np.sqrt(f0_nvalid)

    # Note: all arrays are saved as (defect type, temperature, [run_idx], [bins])
    savedict = {
        "type_names" : DFTYPES,
        "temps" : temps,
        "run_idxs" : run_idxs,
        "free1d_avg" : f1_mean,
        "free1d_std" : f1_std,
        "free1d_nvalid" : f1_nvalid,
        "free1d" : free1d_runs,
        "free0s_avg" : f0_mean,
        "free0s_std" : f0_std,
        "free0s_nvalid" : f0_nvalid,
        "free0s" : free0s_runs,
        "free2d" : free2d_runs,
        "bins1d" : bins,
        "bins2d" : bins2d
    }

    np.savez_compressed(dir_out / "reaction-coord.npz", **savedict)


if __name__ == '__main__':
    main()