import argparse
from icedfmods.Helper_modules import DATA_CACHE

def_keyfile = "CL-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", action = "append", help = "Key file to read data from")

args = parser.parse_args()

import json
import numpy as np
from pathlib import Path
import re

START = 4           # Start of diffusion fit window (ps)
END = 20            # End of diffusion fit window (ps)
MIN_TAU = 50        # The minimum required max lag time (tau) for a run to contribute to diffusion (ps)

def main():
    all_files = args.input_files if len(args.input_files) > 0 else def_keyfile
    for file in all_files:
        print(f"Starting {file}")
        save_diffusion(DATA_CACHE / f"templates/{file}")

def save_diffusion(json_input_file):

    # Load run info
    with open(json_input_file, "r") as f:
        runparams = json.loads(f.read())

    # Constant across all run types
    run_name = runparams["name"]
    DT = runparams["dt"] * 1e-3 # switch to ps

    # Iterating variables
    dftypes = runparams["defect_types"]
    run_idxs = runparams["run_indices"]

    # Use only defect systems
    if "p0m0" in dftypes: dftypes.remove("p0m0")

    # Simulation variables: not the main thing we want to converge, but different cases of it to compare over
    temps = runparams["temperature"]

    # Get input files
    data_dir = Path(runparams["input_dir"])

    # Get output directory
    out_dir = DATA_CACHE / runparams["parent_folder"]
    diff_saveloc = out_dir / "diffusion.npz"

    # All defect types list
    DFTYPES = []

    # Build output lists
    avgs = []
    stds = []
    dfts = []
    types = []

    # Only take contributions from systems we expect to be relevant (maybe change this later?)
    contributors = {
        "OH": ["p0m1"],
        "L": ["p0m1"],
        "H3O": ["p1m0"],
        "D": ["p1m0"]
    }   # BEWARE: WILL BREAK IF ADD MORE THAN 1 EACH DUE TO DIFF_DICT STRUCTURE
    for T in temps:
        for dft in dftypes:
            diff_runs = []
            for run_num in run_idxs:
                # Find parent directory (pxmY-ZZ)
                out_dir_rich = out_dir / f"{dft}-{run_num:02d}"
                
                inputmap = [dft, f"{run_num:02d}", T]
                input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
                dir_in = data_dir / input_formatted

                msd_savepath = out_dir_rich / f'{dir_in.name}-msd.npz'

                msd_dict = np.load(msd_savepath)

                # Initialize DFTYPES if not present, otherwise make sure it is consistent
                if len(DFTYPES) > 0:
                    assert (DFTYPES == msd_dict["type_names"]).all(), f"Wrong defect types {msd_dict["type_names"]} != {DFTYPES}"
                else:
                    DFTYPES = msd_dict["type_names"]

                msd = {name : [] for name in DFTYPES}
                diff = {name : np.nan for name in DFTYPES}
                for name in DFTYPES:
                    mask_by_name = msd_dict["type"] == list(DFTYPES).index(name)

                    # Using masked FFT counts here
                    msd[name], wsum = masked_msd_track_avg(msd_dict["tau"][mask_by_name], msd_dict["msd"][mask_by_name], msd_dict["counts"][mask_by_name])

                    # If we have tracks from defect matching, weight those accordingly
                    #if "track_id" in msd_dict.keys():
                    #    msd[name], wsum = weighted_msd_track_avg(msd_dict["track_id"][mask_by_name], msd_dict["tau"][mask_by_name], msd_dict["msd"][mask_by_name])
                    #else:
                    #    msd[name] = msd_dict["msd"][mask_by_name]

                    # Watch out for empty values
                    if msd[name].size > 0:
                        diff[name] = get_diffusion(msd[name], DT, START, END, MIN_TAU)    # dt is in ps here (and so are start/end/min_tau)

                    # print(f"{dft}-{run_num:02d}-T{T} | {name:5s} | {wsum[int(END/DT)]/int(END/DT) if wsum.size > int(END/DT) else 0:10.4f} | {wsum.size:8d} | {np.unique(msd_dict["track_id"][mask_by_name]).size}")

                diff_runs.append(diff)

            # Get average value and stderr of diffusion coefficient
            for type_code, name in enumerate(DFTYPES):
                if dft in contributors[name]:
                    diff_by_run = [diff_runs[ri][name] for ri in range(len(run_idxs))]
                    Nvalid = len(diff_by_run) - np.count_nonzero(np.isnan(diff_by_run))
                    
                    avgs.append(np.nanmean(diff_by_run, axis = 0))
                    stds.append(np.nanstd(diff_by_run, axis = 0, ddof = 1) / np.sqrt(Nvalid)) # ddof = 1 is Bessel correction (N-1)

                    dfts.append(dft)

                    types.append(type_code)

    # Refactor diff_dict into depth-1 dictionary with boolean lookups
    diff_dict = {
        "dt" : DT,
        "run_name" : run_name,
        "temps" : temps,
        "type_names" : DFTYPES,
        "type_codes" : types,
        "avgs" : avgs,
        "stds" : stds,
        "run_types" : dfts
    }

    # Save diff_dict (will also convert to numpy array which is nice)
    np.savez(diff_saveloc, **diff_dict)

def weighted_msd_track_avg(tids, taus, msds):

    # Get unique track ids
    tids_unique = np.unique(tids)

    if tids_unique.size == 0:
        return np.zeros(0), np.zeros(0)

    # Biggest lag time is longest array
    maxlen = taus.max() + 1
    msd_avg = np.zeros(maxlen)
    wsum = np.zeros(maxlen)

    for tid in tids_unique:

        # Pick out individual tid
        mask_by_tid = tids == tid
        Nf = np.count_nonzero(mask_by_tid)

        # Weight by fraction of time they occupy (longer lived = better stats)
        w = Nf - taus[mask_by_tid]  # Shape (Nf,)

        # Apply weight to relevant msds
        msd_avg[taus[mask_by_tid]] += w * msds[mask_by_tid]

        # Save total weight per entry field
        wsum[taus[mask_by_tid]] += w

    return msd_avg / wsum, wsum

def masked_msd_track_avg(taus, msds, counts):

    if taus.size == 0:
        return np.zeros(0), np.zeros(0)

    # Only take values where we found an MSD between t and t + m
    valid = np.isfinite(msds) * (counts > 0)

    # Set up output arrays
    maxlen = taus.max() + 1
    val = np.zeros(maxlen)
    weight = np.zeros(maxlen)

    # Pool across all tracks (add.at allows overlaps to sum multiple times)
    np.add.at(val, taus[valid], msds[valid] * counts[valid])
    
    # Update weights in the same locations
    np.add.at(weight, taus[valid], counts[valid])

    # Apply weight, leaving in NaN for empty lag times
    weighted_msd = np.full(maxlen, np.nan)
    weighted_msd[weight > 0] = val[weight > 0] / weight[weight > 0]

    return weighted_msd, weight

def get_diffusion(msd, dt, start, end, min_tau):

    #Convert start/end times to index
    start = int(start / dt)
    end = int(end / dt)
    min_tau = int(min_tau / dt)

    Nt = msd.shape[0]
    times = np.arange(Nt) * dt

    # Filter out runs with too crappy statistics
    if Nt < min_tau:
        return np.nan

    # Do fit
    coef = np.polyfit(times[start:end], msd[start:end], 1)

    # Report D in m^2/s (from A^2/ps)
    D = coef[0] * 1e-8 / 6
    
    return D


if __name__ == '__main__':
    main()