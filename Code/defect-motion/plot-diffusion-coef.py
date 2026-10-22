import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = "CL-production.json"
def_figsave = "all-diffusion.svg"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", action = "append", help = "Key file to read data from")
parser.add_argument("-o", "--output", help = "Location to save figure", default = def_figsave)

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

    diff_dicts = []

    all_files = args.input_files if len(args.input_files) > 0 else def_keyfile
    for file in all_files:
        print(f"Loading {file}")
        diff_dicts.append(get_diffusion(DATA_CACHE / f"templates/{file}"))

    print("Making plot")
    fig_savepath = FIGS_CACHE / args.output
    make_plots(diff_dicts, fig_savepath)
    print(f"Figure saved to {fig_savepath}")

def get_diffusion(json_input_file):

    # Load run info
    with open(json_input_file, "r") as f:
        runparams = json.loads(f.read())

    # Get output directory
    out_dir = DATA_CACHE / runparams["parent_folder"]
    diff_saveloc = out_dir / "diffusion.npz"

    return np.load(diff_saveloc)

def make_plots(diff_dicts, fig_savepath):
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

    # Line style properties
    def_markerparams = {'ms' : 12, 'mew' : 3, 'lw' : 3, 'capsize' : 4}
    stylenames = []
    styles = []

    # Create figure and axes
    fig, axs = plt.subplots(2, 2, layout = "constrained", sharey = "row", gridspec_kw={"wspace" : 0, "hspace" : 0.1})
    axs = axs.flatten()
    orientation = np.arange(4).reshape((2,2))
    axnames = ['OH', 'H3O', 'L', 'D']

    # For each run, plot data
    for di, diff_dict in enumerate(diff_dicts):
        DFTYPES = diff_dict["type_names"]
        
        # Get run style from name (water/ice, GGA/hybrid, classical/PIMD)
        stylename, style = lineprops_from_run_name(diff_dict["run_name"])

        # Save style association for legend
        styles.append(style)
        stylenames.append(stylename)
        
        for type_code, (name) in enumerate(DFTYPES):

            # Find appropriate axis
            ax = axs[axnames.index(name)]

            # Filter for correct defect type
            mask_by_name = diff_dict["type_codes"] == type_code

            # Add cap and line style to markerparams
            markerparams = def_markerparams | style

            # Brighten marker center
            markerparams |= {'mfc' : np.clip(clrs[name] * 1.7, 0, 1)}

            # Plot
            ax.errorbar(diff_dict["temps"], np.array(diff_dict["avgs"][mask_by_name])*1e8, yerr = np.array(diff_dict["stds"][mask_by_name])*1e8, c = clrs[name], **markerparams)
            
    # Decorate axes
    for ai, ax in enumerate(axs):
        axname = axnames[ai]

        # Add axis labels on edge axes
        if ai in orientation[:, 0]: ax.set_ylabel(r"$D(T)$" + f" [{unitstring}]")
        if ai in orientation[1, :]: ax.set_xlabel(r"$T$ [K]")

        # Add defect type indicator
        ax.text(0.05, 0.95, dfnametolbl[axname], color = clrs[axname], ha = 'left', va = 'top', transform = ax.transAxes, fontsize = 21)

        # Add legend to D defect axis (least interesting)
        if ai == orientation[1, 1]:
            for si, (style, stylename) in enumerate(zip(styles, stylenames)):
                ax.errorbar([], [], c = 'k', label = stylename, **style)
            
            
            ax.legend()

    # Save figure
    fig_savepath.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(fig_savepath)

def lineprops_from_run_name(run_name):
    # See https://matplotlib.org/stable/api/markers_api.html for ideas

    # Cast to string because .npz wraps in an array
    run_name = str(run_name)
    display_name = ""

    # Determine marker shape from level of theory (alternatively, marker orientation and type can be CL/PI)
    if "hyb" in run_name:
        markershape = "^"
        display_name += "Hybrid "
    else:
        markershape = "s"
        display_name += "GGA "

    # Determine line style from MD type (CL/PI)
    if "PI" in run_name:
        linestyle = "--"
        display_name += "PI "
    else:
        linestyle = "-"
        display_name += "CL "

    # If water, rotate marker
    if "water" in run_name:
        if markershape == "^": markershape = "v"
        elif markershape == "s": markershape = "D"
        display_name += "water"

    return display_name, {'ls' : linestyle, 'marker' : markershape}


if __name__ == '__main__':
    main()