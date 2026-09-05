import argparse

def_keyfile = "./data-cache/bead-convergence.json"
def_plotfile = "./figs-cache/plot-bead-convergence.json"

parser = argparse.ArgumentParser()
parser.add_argument("plot_style", help = "JSON file to read plotting style from", default = def_plotfile, type = str)
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "JSON key file to read dataset locations from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
from pathlib import Path
import numpy as np
import re
from matplotlib import pyplot as plt
from plot_functions import plot_free1d, plot_free2d, plot_barrier, get_keytoorder

# Using kbT = 1 here
prefac = 1

# Get cwd
cwd = Path('.').absolute()

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
name = runparams["name"]
dt = runparams["dt"]

# Iterating variables
dftypes = runparams["defect_types"]     # Here referring to pXmY, the type of defects in my simulaiton
run_idxs = runparams["run_indices"]

# Simulation variables: not the main thing we want to converge, but different cases of it to compare over
temps = runparams["temperature"]
pitypes = runparams["PIMD_type"]

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

inputmap = []
deltas = []
d_sums = []
for dft in dftypes:
    for run_num in run_idxs:
        for pit in pitypes:
            for T in temps:
                for P in nbeads:
                    # Load reaction_coordinates.py output
                    inputs = (dft, f"{run_num:02d}", pit, T, P)
                    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputs)

                    deltas_filein = cwd / f'data-cache/{parent_dir}/{input_formatted}-delta-coord.npz'
                    d_sums_filein = cwd / f'data-cache/{parent_dir}/{input_formatted}-summed-coord.npz'

                    delta_dict = np.load(deltas_filein)
                    d_sum_dict = np.load(d_sums_filein)

                    # Conglomerate same defect types to same column
                    DFTYPES = delta_dict['type_names']
                    assert np.all(d_sum_dict['type_names'] == DFTYPES), f'Defect type mismatch at {input_formatted}: {DFTYPES} != {d_sum_dict['type_name']}'

                    for type_code, name in enumerate(DFTYPES):
                        deltas.append(delta_dict[name])
                        d_sums.append(d_sum_dict[name])
                        inputmap += [(name, dft, run_num, pit, T, P)]

# Reference we can mask to isolate specific variables
inputmap = np.array(inputmap)

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

# Case-specific plotting params (from file)
with open(args.plot_style, "r") as pf:
    plot_style = json.loads(pf.read())

nx, ny = plot_style["layout"]
fig, axs_raw = plt.subplots(*plot_style["layout"], figsize=plot_style["figsize"], gridspec_kw=plot_style["gridspec_kw"])
axs = np.reshape(axs_raw, (nx, ny))

# For multiple plot types in one figure
try:
    styles = plot_style["multistyle"]
    multistyle= True
except(KeyError):
    styles = [plot_style["type"]]
    multistyle= False

# Mapping for item name to its column in inputmap inputmap += [(dft, f"{run_num:02d}", pit, T, P)]
nametomap = {
    "defect_types" : 0, 
    "simulation_types" : 1, 
    "run_indices" : 2, 
    "PIMD_types" : 3, 
    "temperatures" : 4, 
    "num_beads" : 5
}

dfnametofancy = {
    "OH" : r"$OH^-$",
    "H3O" : r"$H_3O^+$",
    "L" : r"$L$",
    "D" : r"$D$"
}

nametolbl = {
    "defect_types" : "{}", 
    "simulation_types" : "{}", 
    "run_indices" : "{:02d}", 
    "PIMD_types" : "{}", 
    "temperatures" : "T = {}", 
    "num_beads" : "P = {}"
}

cmap = "viridis"


# Copy dict over to preserve it
plot_style_full = plot_style.copy()

for style in styles:
    plot_fxn = {
        "free1d" : plot_free1d,
        "free2d" : plot_free2d,
        "barrier" : plot_barrier
    }[style]

    # Index to subplot style
    if multistyle: 
        plot_style = plot_style_full[style].copy() # Index into sub-dict

        # Axes to use in this subplot
        nx, ny = plot_style["layout"]
        xs, ys = plot_style["starting"]
    else:
        xs, ys = 0, 0

    # Frozen variables are constant for each supplot (always list of size nx * ny, needs reshaping)
    frozen = plot_style["frozen"]
    for key in frozen.keys():
        frozen[key] = np.array(frozen[key]).reshape((nx, ny))

    # Isolated variables get a line for each value (always list, arbitrary sizes)
    isolated = plot_style["isolated"]

    # Summed variables are streated as the same run in the histogram (always list, arbitrary size)
    summed = plot_style["summed"]

    for xi in range(nx):
        for yi in range(ny):
            ax = axs[xi + xs, yi + ys]

            # Filter in the specific frozen values we want
            frozen_mask = np.ones(inputmap.shape[0], dtype = bool)
            for key in frozen.keys():
                value = frozen[key][xi, yi]
                has_frozen_val = inputmap[:, nametomap[key]] == str(value)
                frozen_mask *= has_frozen_val

            # Build a starting point for each run
            Nr = np.prod([len(val) for val in isolated.values()])
            run_masks = np.ones((Nr, inputmap.shape[0]), dtype = bool)
            run_names = [{} for _ in range(Nr)]
            run_masks[:, :] *= frozen_mask[None, :]

            # Get individual run masks from isolated vars
            block_size = Nr
            for key in isolated.keys():
                values = isolated[key]
                block_size = block_size // len(values)
                for ri in range(Nr):
                    val = values[(ri // block_size) % len(values)]
                    has_run_val = inputmap[:, nametomap[key]] == str(val)
                    run_masks[ri] *= has_run_val
                    run_names[ri] |= {key : val}
            
            # Build a TRUE starting point for sum mask (since we are multiplying)
            Ns = np.prod([len(val) for val in summed.values()])
            sum_mask = np.ones(inputmap.shape[0], dtype = bool)

            # Get total summed mask
            for key in summed.keys():
                values = summed[key]
                key_mask = np.zeros(inputmap.shape[0], dtype = bool)
                for val in values:
                    # Per key, add up all matching values (OR)
                    key_mask += inputmap[:, nametomap[key]] == str(val)
                # Across keys, only pool matching values (AND), so that smthg like p0m1 run_ind = 19726497 isnt captured
                sum_mask *= key_mask

            # Apply summed mask to run_masks (with AND)
            run_masks[:, :] *= sum_mask[None, :]

            # Now, get histogram across each mask for each run
            runlabels = []

            for ri in range(Nr):
                # Pool values for this run from run_mask
                valid_rows = np.nonzero(run_masks[ri])[0]
                pooled_delta = np.concatenate([deltas[k] for k in valid_rows])
                pooled_d_sum = np.concatenate([d_sums[k] for k in valid_rows])

                # Pool corresponding inputs for some plot_fxns to break apart "summed" variables (must be tiled to match len(deltas))
                pooled_input = np.concatenate([np.tile(inputmap[k], (len(deltas[k]), 1)) for k in valid_rows])

                # Get plottable object
                isostyle = plot_style["isolated_style"]
                plotted = plot_fxn(ax, isostyle, isolated, nametomap, pooled_delta, pooled_d_sum, pooled_input, plot_style["nbins"], prefac, cmap = cmap)

            # Get label for all isolated runs
            DEFAULT_LINESTYLE = {"c" : "k", "ls" : "-"}
            keytoorder = get_keytoorder(isostyle, isolated, cmap)
            for key in isolated.keys():
                lstyle = isostyle[key]   # ls or c
                values = isolated[key]

                for i, (val) in enumerate(sorted(values)):
                    styledict = {lstyle : keytoorder[lstyle][i]}
                    linestyle = DEFAULT_LINESTYLE.copy()
                    linestyle.update(styledict)
                    ax.plot([], [], **linestyle, label = nametolbl[key].format(val), dash_capstyle = 'round')


            # Only show legend if we have per-run params
            if len(isolated.keys()) > 0 and yi == 1:
                ax.legend()

            # Create axis title from frozen
            lbls = []
            for key in sorted(frozen.keys()):
                if key == "defect_types":
                    lbls.append(dfnametofancy[frozen[key][xi, yi]])
                else:
                    lbls.append(nametolbl[key].format(frozen[key][xi, yi]))
            axtitle = ', '.join(lbls)

            if style == "barrier": ax.set_title(axtitle)
            current = ax.get_ylim()
            ax.set_ylim(max([current[0], -0.15]), min([current[1], 2]))

fig.savefig(cwd / f'figs-cache/{plot_style_full["figsave"]}.svg')