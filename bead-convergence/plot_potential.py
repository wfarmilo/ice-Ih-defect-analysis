import argparse

def_keyfile = "./data-cache/bead-convergence.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re
import os

# Get cwd
cwd = Path('.')

# Load input params
run_all = args.all

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
run_name = runparams["name"]
dt = runparams["dt"]

# Iterating variables
dftypes = runparams["defect_types"]
run_idxs = runparams["run_indices"]

# Simulation variables: not the main thing we want to converge, but different cases of it to compare over
temps = runparams["temperature"]
pitypes = runparams["PIMD_type"]

# Convergence variable: the dependent variable we are interested in increasing over
nbeads = runparams["num_beads"]

# Get input files
pdbin_dir = Path(runparams["pdb_input_dir"])
pdb_count = runparams["pdb_atom_count"]
data_dir = Path(runparams["input_dir"])

# Get output directory
out_dir = Path("./data-cache") / runparams["parent_folder"]

# Find the inputs for each run

# List of inputs to out function to run
task_params = []

potentials = np.zeros([len(dftypes), len(run_idxs), len(pitypes), len(temps), len(nbeads)])

# Default plotting params
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
    "figure.figsize": (10, 5),
    "legend.fontsize": 12,
    "mathtext.default": "regular",
})

# Make axes
fig, axs = plt.subplots(1, 2, sharey = "row", figsize = (10, 7), gridspec_kw={'wspace' : 0})

for di, dft in enumerate(dftypes):
    for ri, run_num in enumerate(run_idxs):
        for pii, pit in enumerate(pitypes):
            for ti, T in enumerate(temps):
                for pi, P in enumerate(nbeads):
                    # Load reaction_coordinates.py output
                    inputs = (dft, f"{run_num:02d}", pit, T, P)
                    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputs)

                    # Get potential energy term
                    parent_dir = data_dir / f"{input_formatted}"
                    file_in = parent_dir / f"{parent_dir.name}.out"
                    fulltxt = np.loadtxt(file_in)
                    pot = np.mean(fulltxt[:, -1])

                    potentials[di, ri, pii, ti, pi] = pot

sim_to_pair = {
    "p0m1" : r"$OH^-$/$L$ system",
    "p1m0" : r"$H_3O^+$/$D$ system",
}

# Average over run indices
potentials_avgd = np.mean(potentials[:, :, :, :, :], axis = 1)
potentials_stds = np.std(potentials[:, :, :, :, :], axis = 1) / np.sqrt(len(run_idxs))

for di, dft in enumerate(dftypes):
    ax = axs[di]

    # Subtract off classical potential energy
    potentials_scaled = potentials_avgd[di, :, :, :] - potentials_avgd[di, :, :, 0]
    for pii, pit in enumerate(pitypes):
        for ti, T in enumerate(temps):

            # Plot
            ax.errorbar(nbeads, potentials_scaled[pii, ti, :], yerr = potentials_stds[di, pii, ti, :], label = f'{pit}: T = {T}')

            ax.set_title(sim_to_pair[dft])
            ax.set_xscale('log')
            ax.set_xticks(nbeads)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

            # Label
            ax.set_xlabel(r"P")
            if di == 0: ax.set_ylabel(r"$\langle U - U_{CL} \rangle$")
            ax.legend()

fig.savefig(cwd / f'figs-cache/{run_name}-potential.svg')