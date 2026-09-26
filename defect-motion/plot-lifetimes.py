import argparse

parser = argparse.ArgumentParser()
parser.add_argument("from_file", help = "Key file to read data paths from", type = str)
parser.add_argument('temp', help = "Simulation temperature", type = int)

args = parser.parse_args()

import json
import numpy as np
from icedfmods.Defect_mobility import get_lifetime_distribution
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Collect all data files

# Load input params
T = args.temp

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
name = runparams["name"]
dt = runparams["dt"]

# Variables to iterate over
run_idxs = runparams["run_indices"]
sim_types = runparams["defect_types"]

if "p0m0" in sim_types: sim_types.remove("p0m0")

Nr = len(run_idxs)

# Get input locations
pdbin_dir = Path(runparams["pdb_input_dir"])
data_dir = Path(runparams["input_dir"])

# Get output directory
out_dir = Path("../defect-structure/data-cache") / runparams["parent_folder"]

# Prepare the inputs for each run
DFTYPES = ["OH", "H3O", "L", "D"]
lifetimes = {key : [] for key in DFTYPES}

for dft in sim_types:
    for run_num in run_idxs:
        # Find parent directory (pxmY-ZZ)
        out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

        inputmap = [dft, f"{run_num:02d}", T]
        input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
        dir_in = data_dir / input_formatted

        df_save = out_dir_rich / f'{dir_in.name}-defects.npz'

        assert df_save.exists(), f'Missing input {df_save.absolute()}'

        # Watch out for npz overhead
        with np.load(df_save) as df_dict:

            # Split off clean and defect types
            match dft:
                case "p0m1":
                    for name in ["OH", "L"]:
                        lt = get_lifetime_distribution(df_dict, name)
                        if lt is not None: lifetimes[name].extend(lt)

                case "p1m0":
                    for name in ["H3O", "D"]:
                        lt = get_lifetime_distribution(df_dict, name)
                        if lt is not None: lifetimes[name].extend(lt)

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
num_row = 1
fig, axs = plt.subplots(1, 1, figsize = (8, 6, "cm"))
#axs = axs.reshape((len(DFTYPES), num_row))

DFTYPES_FMT = [r"$OH^-$", r"$H_3O^+$", r"$L$  ", r"$D$  "]
for a_col in range(len(DFTYPES)):
    for a_row in range(num_row):
        ax = axs#[a_col, a_row]

        # Plot clean values
        hist = np.bincount(lifetimes[DFTYPES[a_col]])
        bins = np.arange(len(hist))
        ax.plot(bins * dt / 1000, hist, label = DFTYPES_FMT[a_col])

        # Add defect label
        ax.legend()

        # Set y limits
        ax.set_ylabel(r'counts')
        ax.set_yscale("log")

        # Set x limits
        ax.set_xlabel(r'lifetime (ps)')
        lincut = 3
        ax.set_xscale("symlog", linthresh = lincut, linscale = 2)
        ax.set_xticks([*np.arange(lincut + 1), 10, 100], [*np.arange(lincut + 1), 10, 100])

# Save figure
figdir = Path('../defect-structure/figs-cache')
figname = f'lifetimes-T{T}.svg'
fig.savefig(figdir / figname)