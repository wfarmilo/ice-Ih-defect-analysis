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
out_dir = Path("./data-cache") / runparams["parent_folder"]
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

# Just plot one for now
fig, axs = plt.subplots(2, 2, figsize = (10,10))
axs = axs.flatten()

DFTYPES_FMT = [r"$OH^-$", r"$H_3O^+$", r"$L$", r"$D$"]
for ai in range(axs.size):
    ax = axs[ai]

    # Plot clean values
    ax.plot(rdfs_clean["r_vals"], rdfs_clean["OO"], ls = '--', c = 'C0')
    ax.plot(rdfs_clean["r_vals"], rdfs_clean["OH"], ls = '--', c = 'C1')

    # Plot defect values
    ax.plot(rdfs_by_defect["r_vals"], rdfs_by_defect["OO"][DFTYPES[ai]], ls = '-', c = 'C0')
    ax.plot(rdfs_by_defect["r_vals"], rdfs_by_defect["OH"][DFTYPES[ai]], ls = '-', c = 'C1')

    ax.text(0.5, 0.98, DFTYPES_FMT[ai], ha = 'center', va = 'top', fontsize = 20, transform = ax.transAxes)
    ax.set_xlabel(r'$r [\mathrm{\AA}$]')
    ax.set_ylabel(r'$g(r)$')
    ax.set_ylim([0, 10])

# Set up legend
axs[1].plot([], [], ls = '-', c = 'k', label = f"Clean")
axs[1].plot([], [], ls = '--', c = 'k', label = f"Defect")
axs[1].plot([], [], ls = '-', c = 'C0', label = f"O-O rdf")
axs[1].plot([], [], ls = '-', c = 'C1', label = f"O-H rdf")
axs[1].legend()

# Label figure
fig.text(0.5, 0.98, f'T = {T}K, run {run_num:02d}', ha = 'center', va = 'top', fontsize = 25)

# Save figure
cwd = Path('.')
figdir = cwd / 'figs-cache'
figname = f'rdf-T{T}-{run_num:02d}.svg'
fig.savefig(figdir / figname)