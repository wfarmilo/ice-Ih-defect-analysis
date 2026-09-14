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
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.grid": True,
    "axes.linewidth": 1.25,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "grid.linestyle": "dashed",
    "grid.linewidth": 1,
    "grid.alpha": 0.3,
    "legend.fontsize": 12
})

# Just plot one for now
fig, axs = plt.subplots(len(DFTYPES), 2, figsize = (10, 10), sharex = "col", sharey = "row", gridspec_kw = {"hspace" : 0.05, "wspace" : 0.05})

DFTYPES_FMT = [r"$OH^-$", r"$H_3O^+$", r"$L$  ", r"$D$  "]
for a_row in range(len(DFTYPES)):
    for a_col in range(2):
        ax = axs[a_row, a_col]
        rdftype = ["OO", "OH"][a_col]

        # Plot clean values
        ax.plot(rdfs_clean["r_vals"], rdfs_clean[rdftype], lw = 3, ls = '-', c = 'k')

        # Plot defect values
        ax.plot(rdfs_by_defect["r_vals"], rdfs_by_defect[rdftype][DFTYPES[a_row]], ls = '-', c = 'r')

        if a_col == 0: 
            #ax.text(0.02, 0.98, DFTYPES_FMT[a_row], ha = 'left', va = 'top', transform = ax.transAxes)
            ax.set_ylabel(DFTYPES_FMT[a_row], rotation = "horizontal", ha = "right", fontsize = 20)
        ax.set_ylim([-0.1, 7.95])
        ax.set_yticks(np.arange(0, 7, 2))
        ax.set_xticks(np.arange(0, rdfs_clean["r_vals"].max()))

# Set up legend
axs[0, -1].plot([], [], ls = '-', c = 'k', lw = 3, label = f"Clean")
axs[0, -1].plot([], [], ls = '-', c = 'r', label = f"Defect")
axs[0, -1].legend()

# Add x labels
axs[-1, 0].set_xlabel(r'$r [\mathrm{\AA}$]')
axs[-1, 1].set_xlabel(r'$r [\mathrm{\AA}$]')

# Add column titles
axs[0, 0].set_title(r'O-O rdf', fontsize = 20)
axs[0, 1].set_title(r'O-H rdf', fontsize = 20)

# Label figure
fig.text(0.5, 0.98, f'T = {T}K, run {run_num:02d}', ha = 'center', va = 'top')

# Save figure
figdir = Path('../figs-cache')
figname = f'rdf-T{T}-{run_num:02d}.svg'
fig.savefig(figdir / figname)