import argparse

parser = argparse.ArgumentParser()
parser.add_argument("from_file", help = "Key file to read data paths from", type = str)
parser.add_argument('temp', help = "Simulation temperature", type = int)
parser.add_argument('runidx', help = "Index for simulation temperature", type = int)

args = parser.parse_args()

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Collect all rdfs

# Load input params
T = args.temp
run_num = args.runidx

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
name = runparams["name"]
dt = runparams["dt"]

# Get input locations
pdbin_dir = Path(runparams["pdb_input_dir"])
data_dir = Path(runparams["input_dir"])

# Get output directory
out_dir = Path("../data-cache") / runparams["parent_folder"]
if not(out_dir.exists()): out_dir.mkdir()

# Prepare the inputs for each run
DFTYPES = ["OH", "H3O", "L", "D"]
props = {'r_vals' : None, 'rmax' : None, 'nbins' : None}
rdfs_by_defect = {
     "OO" : {name : [] for name in DFTYPES},
     "OH" : {name : [] for name in DFTYPES},
} | props
rdfs_clean = {
    "OO" : [],
    "OH" : []
}

for dft in ["p0m0", "p0m1", "p1m0"]:
    # Find parent directory (pxmY-ZZ)
    out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    run_name = input_formatted.split("/")[-1]

    rdf_save = out_dir_rich / f'{run_name}-rdf.npz'

    assert rdf_save.exists(), f'Missing input {rdf_save.absolute()}'

    rdf_dict = np.load(rdf_save)

    # Split off clean and defect types
    match dft:
        case "p0m0":
            rdfs_clean["OO"] = rdf_dict["OO"]
            rdfs_clean["OH"] = rdf_dict["OH"]

            # Save overall params (overwritten every time)
            for key in props.keys():
                rdfs_clean[key] = rdf_dict[key]

        case "p0m1":
            for name in ["OH", "L"]:
                rdfs_by_defect["OO"][name] = rdf_dict[f"{name}_O"]
                rdfs_by_defect["OH"][name] = rdf_dict[f"{name}_H"]

            # Save overall params (overwritten every time)
            for key in props.keys():
                rdfs_by_defect[key] = rdf_dict[key]

        case "p1m0":
            for name in ["H3O", "D"]:
                rdfs_by_defect["OO"][name] = rdf_dict[f"{name}_O"]
                rdfs_by_defect["OH"][name] = rdf_dict[f"{name}_H"]

            # Save overall params (overwritten every time)
            for key in props.keys():
                rdfs_by_defect[key] = rdf_dict[key]

# Plotting params
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 21,
    "axes.linewidth": 1.25,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.linewidth": 1,
    "grid.alpha": 0.3,
    "legend.fontsize": 10,
    "figure.constrained_layout.use" : True,
    "lines.dash_capstyle" : "round",
    "mathtext.default" : "regular"
})

# Just plot one for now
fig, axs = plt.subplots(2, len(DFTYPES), figsize = (16, 10, "cm"), sharey = "row", gridspec_kw = {"hspace" : 0.05, "wspace" : 0})

DFTYPES_FMT = [r"$OH^-$", r"$H_3O^+$", r"$L$  ", r"$D$  "]
for a_col in range(len(DFTYPES)):
    for a_row in range(2):
        ax = axs[a_row, a_col]
        rdftype = ["OO", "OH"][a_row]

        # Plot clean values
        ax.plot(rdfs_clean["r_vals"], rdfs_clean[rdftype], lw = 3, ls = '-', c = 'k', alpha = 0.6)

        # Plot defect values
        ax.plot(rdfs_by_defect["r_vals"], rdfs_by_defect[rdftype][DFTYPES[a_col]], ls = '-', c = 'r', lw = 2)

        # Add defect label
        offset = 0 if a_row == 0 else 0.1
        ax.text(0.05 + offset, 0.95, DFTYPES_FMT[a_col], ha = 'left', va = 'top', transform = ax.transAxes)

        # Set y limits
        ax.set_ylim([-0.1, 7.95])
        ax.set_yticks(np.arange(0, 7, 2))
        if a_col == 0: ax.set_ylabel(r'$g_{' + rdftype + r'}(r)$')

        # Set x limits
        xmin = 2 if a_row == 0 else 0.51
        ax.set_xlim((xmin, rdfs_clean["r_vals"].max()))
        ax.set_xticks(np.arange(np.round(xmin), rdfs_clean["r_vals"].max()))
        if a_row == 1: ax.set_xlabel(r'$r [\mathrm{\AA}$]')

# Set up legend
axs[0, -1].plot([], [], ls = '-', c = 'k', lw = 3, label = f"Clean", alpha = 0.6)
axs[0, -1].plot([], [], ls = '-', c = 'r', label = f"Defect")
axs[0, -1].legend(loc = "upper right")

# Save figure
figdir = Path('../figs-cache')
figname = f'rdf-T{T}-{run_num:02d}.svg'
fig.savefig(figdir / figname)