import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = DATA_CACHE / "templates/CL-production.json"
def_savefile = "CL-production/Diffusion-CL.svg"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--from_file", nargs="+", help = "Key file to read data from", default = def_keyfile, type = str)
parser.add_argument("-o", "--output", help = "Name to give figure under FIGS_CACHE", default = def_savefile, type = str)

args = parser.parse_args()

import json
import numpy as np
from pathlib import Path
import re
from matplotlib import pyplot as plt

DFTYPES = ['OH', 'H3O', 'L', 'D']

# Load relevant diffusion coefficients


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

fig, axs = plt.subplots(2, 2, layout = "constrained", sharey = "row", gridspec_kw={"wspace" : 0, "hspace" : 0.1})
axs = axs.flatten()
orientation = np.arange(4).reshape((2,2))
ax_props = [[] for _ in range(axs.size)]

# Add data to axes
for diff_dict in all_diffs:
    for ai, (ax, name) in enumerate(zip(axs, DFTYPES)):
        if name in diff_dict.keys():
            ax.errorbar(diff_dict[temps], np.array(diff_dict[name]["avg"])*1e8, yerr = np.array(diff_dict[name]["std"])*1e8, fmt = '^-', c = clrs[name], **def_markerparams)

            # Add CL/PI, ice/water and GGA/hybrid to legend
            ax_props[ai].extend(diff_dict["run_type"])

# Format axes nicely and add legends
for ai, (ax, name) in enumerate(zip(axs, DFTYPES)):
    # Add axis labels on edge axes
    if ai in orientation[:, 0]: ax.set_ylabel(r"$D(T)$" + f" [{unitstring}]")
    if ai in orientation[1, :]: ax.set_xlabel(r"$T$ [K]")

    # Add defect type indicator
    ax.text(0.05, 0.95, dfnametolbl[name], color = clrs[name], ha = 'left', va = 'top', transform = ax.transAxes, fontsize = 21)

# Save figure
fig_savepath = (FIGS_CACHE / args.output).absolute()
fig_savepath.parent.mkdir(parents = True, exist_ok = True)
fig.savefig(fig_savepath)