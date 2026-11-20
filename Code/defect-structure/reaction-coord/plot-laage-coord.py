import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = "CL-production.json"
def_figsave = "laage-coord-T300.svg"

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
    water = get_laage_data("water-production.json", EXAMPLE_TEMP)
    ice = get_laage_data("CL-production.json", EXAMPLE_TEMP, deltaname="cutoff-delta-coord")

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

    datasets = (water, ice)


    # Get bins
    dmax = max([np.abs(d["delta"]).max() for d in datasets if d is not None])
    dbins = np.linspace(-dmax, dmax, NBINS)
    gbins = np.linspace(-np.pi/2, 3*np.pi/2, NBINS)

    # Get data
    free = []
    for di, data in enumerate(datasets):
        if data is not None:
            free.append(free2d(data["delta"], data["gamma"], EXAMPLE_TEMP, dbins, gbins))
        else: 
            free.append(np.full((NBINS-1, NBINS-1), np.nan))

    
    # Make figure
    fig, axs = plt.subplots(1, 3, figsize = (12, 4), layout = "constrained", width_ratios=(1, 1, 0.2))

    # Partition axes
    ax_2D = axs[:2]

    # Get maximum 2d free energy for color scale
    free2D_limit = np.nanmax(free)


    # Make plot (2D, Laage coord vs delta)
    for di, data in enumerate(datasets):
        ax = ax_2D[di]
        img = plot_2D(ax, free[di], dbins, gbins, colormax = free2D_limit)
        ax.text(0.02, 0.98, ["Water", "Ice"][di], transform = ax.transAxes, ha = 'left', va = 'top')
        ax.set_yticks([-np.pi/4, np.pi/4, 3*np.pi/4, 5*np.pi/4], ["(3a, 4b)", "(3a, 3b)", "(4a, 3b)", "(4a, 4b)"])

    # Add colorbar to last axis
    cax = axs[-1]
    cbar = fig.colorbar(img, cax = cax, label = r"Free energy (kcal mol$^{-1}$)")
    cbar.set_ticks(np.arange(0, free2D_limit + 1e-9, 0.5))

    return fig

def free2d(delta, gamma, T, dbins, gbins):

    # Laage coordinate mirrors different (depends on Oa/Ob def'n)
    gamma_m = np.mod(np.pi - gamma, 2*np.pi) - np.pi/2  # Taking pi/2 - gamma, then wrapping to -pi/2, 3pi/2

    # Apply mirroring now
    delta_all = np.concatenate([delta, -delta])
    gamma_all = np.concatenate([gamma, gamma_m])

    # Get free energy
    hist, _, _ = np.histogram2d(delta_all, gamma_all, bins = (dbins, gbins))
    free = -KB * T * np.log(hist / hist.sum())
    free[~np.isfinite(free)] = np.nan
    return free - np.nanmin(free)

def plot_2D(ax, free2d, dbins, gbins, colormax = None):

    # Get colormap and set scale
    cmap = plt.get_cmap("coolwarm").copy()
    cmax = np.nanmax(free2d) if colormax is None else colormax
    norm2D = plt.Normalize(vmin = 0, vmax = cmax)

    # Plot
    image = ax.imshow(
        free2d.T,
        cmap = cmap,
        norm = norm2D,
        aspect = 'auto',
        origin = 'lower',
        extent = (dbins[0], dbins[-1], gbins[0], gbins[-1])
    )

    return image

def get_laage_data(template_file : str, T : int, deltaname : str = 'delta-coord'):

    DFTYPE = "H3O"
    pooled = {key : [] for key in ["delta", "gamma", "frame", "run", "simtype"]}

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

            rc_laage_in = out_dir_rich / f'{dirin_name}-laage-coord.npy'
            rc_delta_in = out_dir_rich / f'{dirin_name}-{deltaname}.npz'

            frames, gamma = np.load(rc_laage_in)
            delta = np.load(rc_delta_in)[DFTYPE]

            # Orient delta correctly
            delta = -np.abs(delta)

            keep = np.isfinite(gamma) & np.isfinite(delta)


            pooled["delta"].append(delta[keep])
            pooled["gamma"].append(gamma[keep])
            pooled["frame"].append(frames[keep])

            Nk = keep.sum()
            pooled["run"].append(np.full(Nk, run_num))
            pooled["simtype"].append(np.full(Nk, dft))

    return {k: np.concatenate(v) for k,v in pooled.items()}

if __name__ == '__main__':
    main()