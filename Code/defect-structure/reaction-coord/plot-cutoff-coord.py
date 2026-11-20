import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = "CL-production.json"
def_figsave = "cutoff-coord-T300.svg"

parser = argparse.ArgumentParser()
#parser.add_argument("-ff", "--from-file", help = "Key file to read data from", default = def_keyfile)
parser.add_argument("-o", "--output", help = "Location to save figure", default = def_figsave)

args = parser.parse_args()

import json
import numpy as np
from pathlib import Path
import re
from matplotlib import pyplot as plt
import icedfmods.Reaction_coordinates as rc

NBINS = 100

# Choose 1 for units of Kb, 0.0019872041 for kcal/mol 
KB = 0.0019872041

EXAMPLE_TEMP = 300

BJERRUM_NAMES = ["L", "D"]

assert NBINS % 2 == 0, f"NBINS must be even ({NBINS})"

def main():

    print(f"Loading files")
    water = get_delta_data("water-production.json", EXAMPLE_TEMP)
    ice = get_delta_data("CL-production.json", EXAMPLE_TEMP, deltaname="cutoff-delta-coord")

    print("Making plot")
    fig_savepath = FIGS_CACHE / args.output
    fig = make_plots(water, ice)

    fig.savefig(fig_savepath)
    print(f"Figure saved to {fig_savepath}")

def make_plots(water, ice):
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

    datasets_all = (water, ice)

    DFTYPES = list(water.keys() if water is not None else ice.keys())

    # Make figure
    fig, axs_all = plt.subplots(2, 4, figsize = (12, 8), layout = "constrained", width_ratios=(1, 1, 0.2, 1))

    # Get bins
    dmax = np.max([[np.abs(d[name]["delta"]).max() for name in DFTYPES ] for d in datasets_all if d is not None])
    omax = np.max([[np.abs(d[name]["omega"]).max() for name in DFTYPES ] for d in datasets_all if d is not None])
    omin = np.min([[np.abs(d[name]["omega"]).min() for name in DFTYPES ] for d in datasets_all if d is not None])
    dbins = np.linspace(-dmax, dmax, NBINS)
    obins = np.linspace(omin, omax, NBINS)

    for type_code, name in enumerate(DFTYPES):
        axs = axs_all[type_code, :]

        datasets = [d[name] for d in datasets_all if d is not None]

        # Get data
        free2d = []
        free1d = []
        for di, data in enumerate(datasets):
            if data is not None:
                free2d.append(get_free2d(data["delta"], data["omega"], EXAMPLE_TEMP, dbins, obins))
                free1d.append(get_free1d(data["delta"], EXAMPLE_TEMP, dbins, data["run"]))
            else: 
                free2d.append(np.full((NBINS-1, NBINS-1), np.nan))
                free1d.append(np.full((2, NBINS-1), np.nan))


        # Partition axes
        ax_2D = axs[:2]

        # Get maximum 2d free energy for color scale
        free2D_limit = np.nanmax(free2d)


        # Make plot (2D, summed coord vs delta)
        for di, data in enumerate(datasets):
            ax = ax_2D[di]
            img = plot_2D(ax, free2d[di], dbins, obins, colormax = free2D_limit)
            ax.text(0.02, 0.98, ["Water", "Ice"][di], transform = ax.transAxes, ha = 'left', va = 'top')
            ax.text(0.98, 0.98, dfnametolbl[name], transform = ax.transAxes, ha = 'right', va = 'top')

        # Add 1D plots to final axis
        ax_1D = axs[3]

        for di, data in enumerate(datasets):
            plot_1D(ax_1D, free1d[di][0], free1d[di][1], dbins, label = ["Water", "Ice"][di])

        ax_1D.text(0.98, 0.98, dfnametolbl[name], transform = ax_1D.transAxes, ha = 'right', va = 'top')
        ax_1D.legend()
        ax_1D.set_xlim([-0.7, 0.7])
        ax_1D.set_ylim([0, 3])

    # Add colorbar to third axis
    gs = axs_all[0, 2].get_gridspec()
    for ax in axs_all[:,2]:
        ax.remove()
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(img, cax = cax, label = r"Free energy (kcal mol$^{-1}$)")
    cbar.set_ticks(np.arange(0, free2D_limit + 1e-9, 1))

    return fig

def get_free1d(delta_all, T, dbins, runidxs):

    # Partition delta by run index
    runs_unique = np.unique(runidxs)
    free_by_run = []
    for ri in runs_unique:
        mask_by_run = runidxs == ri
        delta = delta_all[mask_by_run]

        # Apply mirroring now
        delta_m = np.concatenate([delta, -delta])

        hist, _ = np.histogram(delta_m, bins = dbins)
        free = -KB * T * np.log(hist / hist.sum())
        free[~np.isfinite(free)] = np.nan
        free -= np.nanmin(free)

        free_by_run.append(free)

    N_valid = np.isfinite(free_by_run).sum(axis = 0)    # How many valid runs per bin
    free1d_avg = np.nanmean(free_by_run, axis = 0)
    free1d_std = np.nanstd(free_by_run, ddof = 1, axis = 0) / np.sqrt(N_valid)


    return free1d_avg, free1d_std

def get_free2d(delta, omega, T, dbins, obins):

    # Apply mirroring now
    delta_all = np.concatenate([delta, -delta])
    omega_all = np.concatenate([omega, omega])

    # Get free energy
    hist, _, _ = np.histogram2d(delta_all, omega_all, bins = (dbins, obins))
    free = -KB * T * np.log(hist / hist.sum())
    free[~np.isfinite(free)] = np.nan
    return free - np.nanmin(free)

def plot_1D(ax, free1d_avg, free1d_std, dbins, label = ''):

    bins_centered = (dbins[1:] + dbins[:-1])/2

    ax.plot(bins_centered, free1d_avg, label = label)
    ax.fill_between(bins_centered, free1d_avg - free1d_std, free1d_avg + free1d_std, alpha = 0.3, linewidth = 0)

    return

def plot_2D(ax, free2d, dbins, obins, colormax = None):

    # Get colormap and set scale
    cmap = plt.get_cmap("viridis").copy()
    cmax = np.nanmax(free2d) if colormax is None else colormax
    norm2D = plt.Normalize(vmin = 0, vmax = cmax)

    # Plot
    image = ax.imshow(
        free2d.T,
        cmap = cmap,
        norm = norm2D,
        aspect = 'auto',
        origin = 'lower',
        extent = (dbins[0], dbins[-1], obins[0], obins[-1])
    )

    return image

def plot_2D_contour(ax, free2d, dbins, obins, colormax = None, step = 0.25):

    # Get colormap and set scale
    cmap = plt.get_cmap("viridis").copy()
    cmax = np.nanmax(free2d) if colormax is None else colormax

    # Set levels step
    levels = np.arange(0, cmax + 1e-9, step)

    # Center bins
    dcent = (dbins[1:] + dbins[:-1])/2
    ocent = (obins[1:] + obins[:-1])/2

    # Plot
    image = ax.contourf(
        dcent,
        ocent,
        free2d.T,
        cmap = cmap,
        levels = levels,
        extend = "max"
    )

    return image

def get_delta_data(template_file : str, T : int, deltaname : str = 'delta-coord'):

    DFTYPES_TO_KEEP = ["OH", "H3O"]
    pooled = {name : {key : [] for key in ["delta", "omega", "run", "simtype"]} for name in DFTYPES_TO_KEEP}

    with open(DATA_CACHE / f"templates/{template_file}", "r") as f:
        runparams = json.loads(f.read())

    out_dir = DATA_CACHE / runparams["parent_folder"]

    # Iterating variables
    dftypes = [d for d in runparams["defect_types"] if d != "p0m0"] # include only defect runs
    run_idxs = runparams["run_indices"]
    assert T in runparams["temperature"], f'T={T} not in template {runparams["name"]} ({runparams["temperature"]})' 

    for dft in dftypes:
        for run_num in run_idxs:
                out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

                inputmap = [dft, f"{run_num:02d}", T]
                input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
                dirin_name = input_formatted.split('/')[-1]

                rc_delta_in = out_dir_rich / f'{dirin_name}-{deltaname}.npz'
                rc_omega_in = out_dir_rich / f'{dirin_name}-{deltaname.replace("delta", "summed")}.npz'

                for name in DFTYPES_TO_KEEP:

                    delta = np.load(rc_delta_in)[name]
                    omega = np.load(rc_omega_in)[name]

                    # Orient delta correctly
                    delta = -np.abs(delta)

                    keep = np.isfinite(delta)

                    pooled[name]["delta"].append(delta[keep])
                    pooled[name]["omega"].append(omega[keep])

                    Nk = keep.sum()
                    pooled[name]["run"].append(np.full(Nk, run_num))
                    pooled[name]["simtype"].append(np.full(Nk, dft))

    # Concatenate into 1D arrays (cheaper than using .extend everywhere)
    pooled_linear = {name : {k: np.concatenate(v) for k,v in pooled[name].items()} for name in DFTYPES_TO_KEEP}
    return pooled_linear

if __name__ == '__main__':
    main()