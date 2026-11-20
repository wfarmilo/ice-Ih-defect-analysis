import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = "CL-production.json"
def_figsave = "reaction-coord.svg"

parser = argparse.ArgumentParser()
parser.add_argument("-ff", "--from-file", help = "Key file to read data from", default = def_keyfile)
parser.add_argument("-o", "--output", help = "Location to save figure", default = def_figsave)

args = parser.parse_args()

import json
import numpy as np
from pathlib import Path
import re
from matplotlib import pyplot as plt
import icedfmods.Reaction_coordinates as rc

NBINS = 100
KB = 1

EXAMPLE_TEMP = 275

BJERRUM_NAMES = ["L", "D"]

assert NBINS % 2 == 0, f"NBINS must be even ({NBINS})"

def main():

    file = args.from_file

    print(f"Loading {file}")
    with open(DATA_CACHE / f"templates/{file}", "r") as f:
        runparams = json.loads(f.read())
    data = np.load(DATA_CACHE / f"{runparams["parent_folder"]}/reaction-coord.npz")

    print("Making plot")
    fig_savepath = FIGS_CACHE / args.output
    fig = make_plots(data)

    fig.savefig(fig_savepath)
    print(f"Figure saved to {fig_savepath}")

def make_plots(data):
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

    # Global properties
    DFTYPES = data["type_names"]
    temps = data["temps"]

    # Make figure
    fig, axs = plt.subplots(2, 6, figsize = (24, 8), layout = "constrained")

    # Partition axes
    ax_2D = axs[:, :2].flatten()
    ax_F0 = axs[:, 2:4].flatten()
    ax_1D = axs[:, 4:6].flatten()

    T_2D_index = list(temps).index(EXAMPLE_TEMP)

    # Get maximum 2d free energy for color scale
    free2D_limit = np.nanmax(data["free2d"][:, T_2D_index, :, :])

    # Make all plots
    for type_code, name in enumerate(DFTYPES):
        # Classify defect for limits and such
        defect_category = "bjerrum" if name in ["L", "D"] else "ionic"

        # Make plot (2D)
        ax = ax_2D[type_code]
        plot_2D(ax, data["free2d"][type_code, T_2D_index, :, :], data["bins2d"][type_code, T_2D_index, :, :], colormax = free2D_limit)
        ax.text(0.98, 0.98, dfnametolbl[name], transform = ax.transAxes, ha = 'right', va = 'top')

        # Switch to degrees
        if name in BJERRUM_NAMES:
            angticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
            ax.set_yticks(angticks, [f"{tick * 180 / np.pi:.0f}" for tick in angticks])

        # Make plot (F0)
        ax = ax_F0[type_code]
        plot_F0(ax, data["free0s_avg"][type_code, :], data["free0s_std"][type_code, :], temps)
        ax.text(0.98, 0.98, dfnametolbl[name], transform = ax.transAxes, ha = 'right', va = 'top')

        # Print f0 data
        print(f"For {name} defect:")
        print_F0(data["free0s_avg"][type_code, :], temps)

        # Make plot (1D reaction coord)
        ax = ax_1D[type_code]
        plot_1D(ax, data["free1d_avg"][type_code, :], data["free1d_std"][type_code, :], temps, data["bins1d"][type_code, :])
        ax.text(0.98, 0.98, dfnametolbl[name], transform = ax.transAxes, ha = 'right', va = 'top')
        lims = [-1.2, 1.2] if name in BJERRUM_NAMES else [-0.5, 0.5]
        ax.set_xlim(lims)
        ax.set_ylim(0, 1000)

    return fig

def plot_2D(ax, free2d, bins2d, colormax = None):

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
        extent = (bins2d[0][0], bins2d[0][-1], bins2d[1][0], bins2d[1][-1])
    )

    # TODO: make this figure-wide?
    plt.colorbar(image, ax = ax)

    return

def plot_F0(ax, f0_avg, f0_std, temps):
    ax.errorbar(temps, f0_avg, yerr = f0_std, capsize = 3, marker = 'o')
    ax.set_xticks(temps)
    return

def print_F0(f0_avg, temps):
    
    print(" T (K) | F(0) kB | F(0) kcal/mol")

    KB_VAL = 1.38e-23
    MOL_VAL = 6.022e23
    JOULE_PER_KCAL = 4.184 * 1000
    
    
    for ti, T in enumerate(temps):
        print(f" {T:5d} | {f0_avg[ti]:7.2f} | {f0_avg[ti] * KB_VAL * MOL_VAL / JOULE_PER_KCAL:.2f}")
    return

def plot_1D(ax, f1d_avg, f1d_std, temps, bins):
    # Index colormap for temperature runs
    cmap = plt.get_cmap("viridis").copy()
    colors = cmap(np.linspace(0, 0.8, len(temps)))

    # Center bins
    bins_centered = (bins[1:] + bins[:-1])/2

    # Plot each temp
    for ti, T in enumerate(temps):
        avg = f1d_avg[ti]
        std = f1d_std[ti]
        ax.plot(bins_centered, avg, label = f"T = {T}", c = colors[ti])
        ax.fill_between(bins_centered, avg - std, avg + std, color = colors[ti], alpha = 0.3, linewidth = 0)


    ax.legend()

    return


if __name__ == '__main__':
    main()