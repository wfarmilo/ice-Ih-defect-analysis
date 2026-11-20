import argparse
from icedfmods.Helper_modules import DATA_CACHE


def_keyfile = "CL-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)
parser.add_argument("-dfs", "--defects", nargs = "*", help = "Which defects to analyze (forces update mode on them)", default = False)
parser.add_argument("-ag", "--aggregate-only", action = "store_true", help = "Only perform the final aggregation and free energy computation step")
parser.add_argument("-hbc", "--hydrogen_bonds_cutoff", action = "store_true", help = "Only get defect hydrogen indices")

args = parser.parse_args()

import json
from icedfmods import Reaction_coordinates as rc, Cutoff_HBN as cHBN
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pathlib import Path
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_TASKS = 8

def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, T, runparams, run_all, run_defects, run_hbc):

    # Get home directory from run params
    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    # Input files
    traj_file = dir_in / f'traj-{dir_in.name}.dcd'
    hbn_in = out_dir_rich / f'{dir_in.name}-HBN.npz'
    chbn_in = out_dir_rich / f'{dir_in.name}-cutoff-HBN.npz'
    dfs_in = out_dir_rich / f'{dir_in.name}-defects.npz'

    # Output files
    rc_delta_out = out_dir_rich / f'{dir_in.name}-delta-coord.npz'
    rc_d_sum_out = out_dir_rich / f'{dir_in.name}-summed-coord.npz'
    rc_hydid_out = out_dir_rich / f'{dir_in.name}-defect-hydidx.npz'
    rc_cdelta_out = out_dir_rich / f'{dir_in.name}-cutoff-delta-coord.npz'
    rc_csum_out   = out_dir_rich / f'{dir_in.name}-cutoff-summed-coord.npz'

    # If in update mode and any file found, skip it
    if not(run_defects) and not(run_all) and rc_delta_out.exists() and rc_d_sum_out.exists() and rc_hydid_out.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # If looking for a specific defect update, make sure the file exists
    if run_defects:
        assert rc_delta_out.exists() and rc_d_sum_out.exists(), "Cannot run in defect-specific update mode without an existing file"

    # Callout before loading files
    print(f'Reading {dft}-{run_num:02d}/{dir_in.name}')

    # Set up simulation
    u = mda.Universe(pdbin.absolute(), traj_file.absolute(), format = 'dcd')
    u.dimensions = cell_dims
    u.trajectory.add_transformations(trans.wrap(u.atoms, compound = 'atoms'))

    # Callout after dcd not supported spam
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')

    hbn_dict = dict(np.load(hbn_in))
    dfs_dict = dict(np.load(dfs_in))

    tis = hbn_dict["tis"]

    # Minimal loading of big array
    with np.load(chbn_in) as zipped:
        hbonds = {key: zipped[key] for key in ("frame", "donor", "accep", "hyd")}

    # Here referring to OH/H3O/L/D
    DFTYPES = dfs_dict['type_names']
    deltas = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])
    d_sums = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])
    hydidxs = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])
    cdeltas = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])
    cd_sums = {'type_names' : DFTYPES} | dict([(name, []) for name in DFTYPES])

    # Which defects to track
    dfs_to_modify = run_defects if run_defects else dfs_dict['type_names']

    # For hbc run, we only need H3O/OH
    if run_hbc:
        dfs_to_modify = [d for d in dfs_to_modify if d not in ["L", "D"]]

    for name in dfs_to_modify:

        # Get type code from full list
        type_code = list(DFTYPES).index(name)

        # Find atoms of the relevant defect type (OH/H3O/L/D)
        oftype = dfs_dict['type'] == type_code
        frames, atom1s, atom2s = dfs_dict["frame"][oftype], dfs_dict["atom1"][oftype], dfs_dict["atom2"][oftype]

        # For D defects, find next atom1 for each track
        if name == "D": 
            next_atom1s = rc.next_D_by_track(frames, dfs_dict["track_id"][oftype], atom1s)
            prev_atom1s = rc.next_D_by_track(-frames, dfs_dict["track_id"][oftype], atom1s)

        # Get hydidxs and cutoff HBN stuff
        coords, sums, hydidx = cHBN.get_delta_coordinate(u, hbonds, atom1s, frames)

        # Filter out empties
        valid = np.isfinite(coords) & np.isfinite(sums)
        saved_frames = frames[valid]
        hydidxs[name] = np.stack([saved_frames, hydidx[valid]])
        cdeltas[name] = coords[valid]
        cd_sums[name] = sums[valid]

        if not(run_hbc):
            # For L defects, use both atoms for dfidxs
            zipped = zip(frames, atom1s, atom2s)
            for step, (ts, a1, a2) in enumerate(zipped):
                u.trajectory[ts]

                # Unshift equilibrium time to find index of hbn
                ti = ts - tis[0]

                # Stack hydrogen bond network into the relevant row (kind of legacy code here)
                hbn_row = np.stack([hbn_dict["hyd"][ti], hbn_dict["donor"][ti], hbn_dict["accep"][ti], hbn_dict["HBN_angs"][ti]])

                # Stack atoms here as well (note that -1 padding will prevent ionic defects from breaking)
                dfs = np.array([a1, a2], dtype = int)

                # Get delta and sum value
                if name in ["OH", "H3O"]:
                    delta, d_sum = rc.get_ionic_delta(u, dfs, hbn_row)
                elif name in ["L"]:
                    delta, d_sum = rc.get_L_delta(u, dfs, hbn_row)
                elif name in ["D"]:
                    delta, d_sum = rc.get_D_delta_atom(u, ti, dfs, [prev_atom1s[step], next_atom1s[step]], hbn_dict)
                else:
                    raise Exception(f"Invalid atom name {name}")

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
        hydid_cached = np.load(rc_hydid_out)
        cdelta_cached = np.load(rc_cdelta_out)
        cd_sum_cached = np.load(rc_csum_out)

        # Pull unmodified defects from previously saved values
        for key in not_touched:
            deltas[key] = delta_cached[key]
            d_sums[key] = d_sum_cached[key]
            hydidxs[key] = hydid_cached[key]
            cdeltas[key] = cdelta_cached[key]
            cd_sums[key] = cd_sum_cached[key]

    # Save HBC hydidxs
    np.savez_compressed(rc_hydid_out, **hydidxs)
    np.savez_compressed(rc_cdelta_out, **cdeltas)
    np.savez_compressed(rc_csum_out, **cd_sums)

    # If doing normal run, also save regular coords
    if run_hbc:
        return f'{dft}-{run_num:02d}/{dir_in.name} HBC completed'
    else:
        np.savez_compressed(rc_delta_out, **deltas)
        np.savez_compressed(rc_d_sum_out, **d_sums)

        # Result is the diagnostic for printing from the executor
        return f'{dft}-{run_num:02d}/{dir_in.name} completed'

def main():

    # Load run specifications
    run_all = args.all
    run_defects = args.defects
    run_hbc = args.hydrogen_bonds_cutoff

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
                task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all, run_defects, run_hbc))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(temps)])
    max_workers = np.min([tot_workers, MAX_TASKS])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())

    print("Execution complete! Starting aggregation")
    aggregate_files(DATA_CACHE / f"templates/{args.from_file}", out_dir)
    print(f"Aggregation complete, saving to {out_dir / 'reaction_coord.npz'}")

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


