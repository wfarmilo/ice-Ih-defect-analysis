"""
Script to plot the output of get-total-dipole.py so I can see if
my systems are polarizing themselves (and if it is affected by system size)
"""

import argparse
from icedfmods.Helper_modules import DATA_CACHE, FIGS_CACHE

def_keyfile = DATA_CACHE / "templates/CL-production.json"

parser = argparse.ArgumentParser()

parser.add_argument("from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()



import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Collect all dipoles

# Load input params

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
name = runparams["name"]
dt = runparams["dt"]

# Iterating variables
dftypes = runparams["defect_types"]
run_idxs = runparams["run_indices"]

# Simulation variables: not the main thing we want to converge, but different cases of it to compare over
temps = runparams["temperature"]

# Get input locations
pdbin_dir = Path(runparams["pdb_input_dir"])
data_dir = Path(runparams["input_dir"])

# Get output directory for data
out_dir = DATA_CACHE / runparams["parent_folder"]

# Lookup for data masking
data = np.zeros([len(dftypes), len(run_idxs), len(temps)], dtype = object)
norm = np.zeros([len(dftypes), len(run_idxs), len(temps)], dtype = object)
Nt = np.inf
Nt_new = np.inf

for di, dft in enumerate(dftypes):
    for ri, run_num in enumerate(run_idxs):
        # Find parent directory (pxmY-ZZ)
        out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

        # Update max run length
        Nt = np.min([Nt, Nt_new])

        # Continue iterating over dependents
        for Ti, T in enumerate(temps):
            inputmap = [dft, f"{run_num:02d}", T]
            input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
            run_name = input_formatted.split("/")[-1]

            dipole_save = out_dir_rich / f'{run_name}-dipole.npy'

            assert dipole_save.exists(), f'File {dipole_save.absolute()} not found'

            dipole = np.load(dipole_save)

            # Save dipole array in proper spot
            Nt_new = dipole.shape[0]
            data[di, ri, Ti] = dipole
            norm[di, ri, Ti] = np.linalg.norm(dipole, axis = -1)


# Plot properties
errorbar_qty = 20
markerparams = {'lw' : 3, 'capsize' : 4, 'ms' : 12, 'mew' : 3}
lineparams = {
    "p0m0" : {'ls' : '-', 'c' : 'k'},
    "p0m1" : {'ls' : '-', 'c' : 'C0'},
    "p1m0" : {'ls' : '-', 'c' : 'C1'}
}
runlabels = {
    "p0m0" : "Clean",
    "p0m1" : r"$OH^-/L$ system",
    "p1m0" : r"$H_3O^+/D$ system"
}

# Make plots
fig, axs = plt.subplots(len(temps), 1, figsize = (10,10))

for ai, ax in enumerate(axs.flatten()):
    for di, dft in enumerate(dftypes):
        for ri, run_num in enumerate(run_idxs[:1]):
            ax.plot(norm[di, ri, ai] - norm[di, ri, ai][0], **lineparams[dft])
        ax.plot([], [], **lineparams[dft], label = runlabels[dft])
    ax.set_title(f'T = {temps[ai]}K')
    ax.legend()

# Save figure
fig_savefolder = FIGS_CACHE
(fig_savefolder / runparams["parent_folder"]).mkdir(parents = True, exist_ok = True)
fig.savefig(fig_savefolder / f'{runparams["parent_folder"]}/{name}-central-dipole.svg')