import argparse

def_keyfile = "./data-cache/bead-convergence.json"
def_plotfile = "./figs-cache/plot-bead-convergence.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "JSON key file to read dataset locations from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
from pathlib import Path
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib as mpl

# Get cwd
cwd = Path('.').absolute()

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
run_name = runparams["name"]
dt = runparams["dt"]

# Iterating variables
dftypes = runparams["defect_types"]     # Here referring to pXmY, the type of defects in my simulaiton
run_idxs = runparams["run_indices"]

# Simulation variables: not the main thing we want to converge, but different cases of it to compare over
temps = runparams["temperature"]
pitype = runparams["PIMD_type"]

# Convergence variable: the dependent variable we are interested in increasing over
nbeads = runparams["num_beads"]

# Get input files
pdbin_dir = Path(runparams["pdb_input_dir"])
data_dir = Path(runparams["input_dir"])
inputdir_raw = runparams["input_dir"]

# Get output directory
parent_dir = runparams["parent_folder"]
out_dir = Path("./data-cache") / parent_dir
assert out_dir.exists(), f"Directory {out_dir.absolute()} not found. Running from {Path(".").absolute()}"

DFTYPES = ["OH", "H3O"]#, "L", "D"]
PITYPES = ["PILE", "ECON"]
ATTYPES = ["O_", "H_"]
ST = "_std"

ALLTYPES = [at + df for at in ATTYPES for df in DFTYPES]
ALLTYPES_ST = [at + df + ST for at in ATTYPES for df in DFTYPES]

assert (np.array(pitype) == np.array(PITYPES)).all()

inputmap = []
# Nesting structure: pimd_type ["PILE", "ECON"] -> combination of atom type (or std) ["O", "H"], defect name ["OH", "H3O", "L", "D"]
means = {piname : {name: [] for name in ALLTYPES} for piname in PITYPES}
stders = {piname : {name  : [] for name in ALLTYPES} for piname in PITYPES}
bulk = {piname: {name : [] for name in ["O", "H", "O" + ST, "H" + ST]} for piname in PITYPES}
for dft in dftypes:
    for run_num in run_idxs:
        for pit in PITYPES:
            for T in temps:
                for P in nbeads:
                    # Load reaction_coordinates.py output
                    inputs = (dft, f"{run_num:02d}", pit, T, P)
                    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputs)

                    gyr_filein = cwd / f'data-cache/{parent_dir}/{input_formatted}-gyration-radius.npz'

                    gyr_dict = np.load(gyr_filein)

                    # Conglomerate same defect types to same column
                    #assert (gyr_dict['type_names'] == DFTYPES).all()

                    # Save bulk properties for reference
                    for bk in ["O", "H"]: 
                        for st in ["", ST]:
                            bulk[pit][bk + st].append(gyr_dict[bk + "_bulk" + st])

                    for key in ALLTYPES:
                        # Save individual defect types in one big dict, with indexes by atom type
                        means[pit][key].append(gyr_dict[key])
                        stders[pit][key].append(gyr_dict[key + ST])

                    # Lookup map for defect list
                    inputmap += [(pit, dft, run_num, T, P)]

# Reference we can mask to isolate specific variables
inputmap = np.array(inputmap)

# numpy arrays are better
means = {pit : {name: np.array(vals) for name, vals in means[pit].items()} for pit in PITYPES}
stders = {pit : {name: np.array(vals) for name, vals in stders[pit].items()} for pit in PITYPES}
bulk = {pit : {lbl: np.array(vals) for lbl, vals in bulk[pit].items()} for pit in PITYPES}

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

# Defect-based coloring
clrs = {
    "OH" : "C0",     #np.array([221, 48, 37])/255, 
    "H3O" : "C1",    #np.array([145, 20, 73])/255, 
    "L" : "C2",      #np.array([41, 52, 122])/255, 
    "D" : "C3",      #np.array([104, 31, 177])/255,
    "bulk": "k"      #np.array([0,0,0])/255
}

# Mapping for item name to its column in inputmap for code clarity inputmap += [(dft, f"{run_num:02d}", pit, T, P)]
ntm = {
    "pimd_type" : 0,
    "simulation_types" : 1, 
    "run_indices" : 2, 
    "temperatures" : 3, 
    "num_beads" : 4
}

fig, axs = plt.subplots(2, 2, figsize = (12, 12), sharey="row", gridspec_kw={'wspace' : 0})
axs = axs.flatten()

# Get each bead run
beads_unique = np.unique(inputmap[:, ntm["num_beads"]].astype(int))
total_mask = {pit : np.zeros((len(beads_unique), inputmap.shape[0] // len(PITYPES)), dtype = bool) for pit in PITYPES}

for bidx, (bead) in enumerate(beads_unique):
    for pit in PITYPES:
        total_mask[pit][bidx] = (inputmap[:, ntm["num_beads"]] == str(bead))[inputmap[:, ntm["pimd_type"]] == pit]


# Default params for errorbar plots
markerparams = {key : {'lw' : 3, 'capsize' : 4, 'ms' : 12, 'mec' : clrs[key], 'mew' : 3} for key in clrs.keys()}

# Parameters for each axis
AXIS_PARAMS = [
    ["H", "PILE"],
    ["H", "ECON"],
    ["O", "PILE"],
    ["O", "ECON"]
]

for ai, ap in enumerate(AXIS_PARAMS):

    # Get relevant atom type and pimd type
    att = ap[0]
    pit = ap[1]

    # Get axis
    ax = axs[ai]

    # Get individual run data (save myself the indexing headache)
    bk = bulk[pit]
    mask_by_bead = total_mask[pit]

    # Pool by bead idx
    pooled_bulk = {lbl : [np.sum(bk[lbl][mbb])/np.count_nonzero(mbb) for mbb in mask_by_bead] for lbl in [att, att + ST]}

    ax.errorbar(beads_unique, np.sqrt(pooled_bulk[att]), yerr = np.sqrt(pooled_bulk[att + ST]), c = 'k', ls = '--', label = f"Bulk {att}", **markerparams["bulk"])

    for name in DFTYPES:
        # Turn name into key
        key = att + '_' + name

        # Get specific means/stds
        mns = means[pit][key]
        sts = stders[pit][key]

        # Make sure we have a valid value
        valid = mns >= 0

        # Pool by bead idx
        pooled_means = [np.sum(mns[valid * mbb])/np.count_nonzero(valid * mbb) for mbb in mask_by_bead]
        pooled_stder = [np.sum(sts[valid * mbb])/np.count_nonzero(valid * mbb) for mbb in mask_by_bead]

        # Plot defects
        ax.errorbar(beads_unique, np.sqrt(pooled_means), yerr = np.sqrt(pooled_stder), c = clrs[name], label = f"Defect {name}", **markerparams[name])

    ax.text(0.03, 0.97, f"({['a', 'b', 'c', 'd'][ai]}) {pit}: {att}", ha = 'left', va = 'top', transform = ax.transAxes)#, bbox = {'boxstyle' : 'square', 'color' : 'w', 'ec' : 'k'})
    if pit == "PILE": ax.set_ylabel(r"$\mathcal{R}_g\mathrm{ (\AA)}$")
    if att == "O": ax.set_xlabel(r"P")

    ax.legend()
    ax.set_xscale('log')
    ax.set_xticks(beads_unique)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    current = ax.get_ylim()
    ax.set_ylim(current[0], max([current[1], 0.09]))

fig.savefig(cwd / f'figs-cache/ionic-{run_name}-gyration.svg')