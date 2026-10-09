import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = DATA_CACHE / "templates/CL-production.json"
def_savefile = FIGS_CACHE / "CL-production/Diffusion-CL.svg"

parser = argparse.ArgumentParser()
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)
parser.add_argument("-o", "--output", help = "Where to save figure", default = def_savefile, type = str)

args = parser.parse_args()

import json
import numpy as np
from pathlib import Path
import re
from matplotlib import pyplot as plt

START = 4           # Start of diffusion fit window (ps)
END = 20            # End of diffusion fit window (ps)
MIN_TAU = 50        # The minimum required max lag time (tau) for a run to contribute to diffusion (ps)

def main():

    # Load input params
    fig_savepath = Path(args.output).absolute()

    # Load run info
    with open(args.from_file, "r") as f:
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

    # Prepare the inputs for each run

    # Defect matching parameters
    LIFETIME = runparams["defect_lifetime"]
    RMAX = runparams["defect_maxjump"]

    # All defect types, in the order as presented in Defect_tracking.py
    DFTYPES = np.array(['OH', 'H3O', 'L', 'D'])
    
    # Build output dict
    diff_dict = {name : {
                        "avg" : [], 
                        "std" : [], 
                        "dft" : []
                        } for name in DFTYPES}

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

                assert (DFTYPES == msd_dict["type_names"]).all(), f"Wrong defect types {msd_dict["type_names"]} != {DFTYPES}"
                msd = {name : [] for name in DFTYPES}
                diff = {name : np.nan for name in DFTYPES}
                for name in DFTYPES:
                    mask_by_name = msd_dict["type"] == list(DFTYPES).index(name)
                    msd[name], wsum = weighted_msd_track_avg(msd_dict["track_id"][mask_by_name], msd_dict["tau"][mask_by_name], msd_dict["msd"][mask_by_name])

                    # Watch out for empty values
                    if msd[name].size > 0:
                        diff[name] = get_diffusion(msd[name], DT, START, END, MIN_TAU)    # dt is in ps here (and so are start/end/min_tau)

                    # print(f"{dft}-{run_num:02d}-T{T} | {name:5s} | {wsum[int(END/DT)]/int(END/DT) if wsum.size > int(END/DT) else 0:10.4f} | {wsum.size:8d} | {np.unique(msd_dict["track_id"][mask_by_name]).size}")

                diff_runs.append(diff)

            # Get average value and stderr of diffusion coefficient
            for name in DFTYPES:
                if dft in contributors[name]:
                    diff_by_run = [diff_runs[ri][name] for ri in range(len(run_idxs))]
                    Nvalid = len(diff_by_run) - np.count_nonzero(np.isnan(diff_by_run))
                    
                    diff_dict[name]["avg"].append(np.nanmean(diff_by_run, axis = 0))
                    diff_dict[name]["std"].append(np.nanstd(diff_by_run, axis = 0) / np.sqrt(Nvalid))

                    diff_dict[name]["dft"].append(dft)

    #################### Plotting time ##########################################
    #Plot params
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 21,
        "axes.grid": True,
        "axes.linewidth": 1.25,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 1.25,
        "ytick.minor.width": 1.25,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "grid.linestyle": "dashed",
        "grid.linewidth": 1,
        "grid.alpha": 0.3,
        "figure.figsize": (10, 8),
        "legend.fontsize": 12,
        "mathtext.default": "regular",
    })

    unitstring = r'$\times 10^{-8}$ m$^2$s$^{-1}$'

    clrs = {
        "OH" : np.array([221, 48, 37])/255, 
        "L" : np.array([41, 52, 122])/255, 
        "H3O" : np.array([145, 20, 73])/255, 
        "D" : np.array([104, 31, 177])/255
    }

    dfnametolbl = {
        "OH" : r"$OH^-$",
        "L" : r"$L$", 
        "H3O" : r"$H_3O^+$", 
        "D" : r"$D$"
    }

    def_markerparams = {'ms' : 12, 'mew' : 3, 'lw' : 3, 'capsize' : 4}

    fig, axs = plt.subplots(2, 2, gridspec_kw={"wspace" : 0.15, "hspace" : 0.1, "left" : 0.15, "bottom" : 0.15})
    axs = axs.flatten()
    for ax, name in zip(axs, DFTYPES):
        ax.errorbar(temps, np.array(diff_dict[name]["avg"])*1e8, yerr = np.array(diff_dict[name]["std"])*1e8, fmt = '^-', c = clrs[name], **def_markerparams)
        ax.set_ylabel(r"$D(T)$" + f" [{unitstring}]")
        ax.set_xlabel(r"$T$ [K]")
        ax.text(0.05, 0.95, dfnametolbl[name], color = clrs[name], ha = 'left', va = 'top', transform = ax.transAxes, fontsize = 21)

    fig_savepath.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(fig_savepath)


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