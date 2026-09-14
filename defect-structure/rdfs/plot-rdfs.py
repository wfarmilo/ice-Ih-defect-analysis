import argparse

def_keyfile = "../data-cache/CL-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Collect all rdfs

# Load input params
run_all = args.all

# Load run info
with open(args.from_file, "r") as f:
    runparams = json.loads(f.read())

# Constant across all run types
name = runparams["name"]
dt = runparams["dt"]

# Iterating variables
dftypes = runparams["defect_types"]

dftypes_clean = ["p0m0"]
assert dftypes_clean[0] in dftypes, f"Defect type p0m0 not found in {dftypes}"

run_idxs = runparams["run_indices"]

# Simulation variables: not the main thing we want to converge, but different cases of it to compare over
temps = runparams["temperature"]

# Get input locations
pdbin_dir = Path(runparams["pdb_input_dir"])
data_dir = Path(runparams["input_dir"])

# Get output directory
out_dir = Path("../data-cache") / runparams["parent_folder"]

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
input_map = {"df" : [], "cl" : []}

for dft in dftypes:
    for run_num in run_idxs:
        # Find parent directory (pxmY-ZZ)
        out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

        # Continue iterating over dependents
        for T in temps:
                inputmap = [dft, f"{run_num:02d}", T]
                input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
                run_name = input_formatted.split("/")[-1]

                rdf_save = out_dir_rich / f'{run_name}-rdf.npz'

                assert rdf_save.exists(), f'Missing input {rdf_save.absolute()}'

                rdf_dict = np.load(rdf_save)

                # Split off clean and defect types
                if dft in dftypes_clean:
                    rdfs_clean["OO"].append(rdf_dict["OO"])
                    rdfs_clean["OH"].append(rdf_dict["OH"])

                    # Add to clean lookup
                    input_map["cl"].append((dft, run_num, T))

                    # Save overall params (overwritten every time)
                    for key in props.keys():
                        rdfs_clean[key] = rdf_dict[key]
                else:
                    assert (rdf_dict["DFTYPES"] == np.array(DFTYPES)).all()
                    for name in DFTYPES:
                        rdfs_by_defect["OO"][name].append(rdf_dict[f"{name}_O"])
                        rdfs_by_defect["OH"][name].append(rdf_dict[f"{name}_H"])

                    # Add to defect lookup
                    input_map["df"].append((dft, run_num, T))

                    # Save overall params (overwritten every time)
                    for key in props.keys():
                        rdfs_clean[key] = rdf_dict[key]

# Convert to arrays
for key in input_map.keys():
    input_map[key] = np.array(input_map[key], dtype = str)

# Convenience dicts
NAMETOINP = {
     "simulation_type" : 0,
     "run_index" : 1,
     "temp" : 2
}
