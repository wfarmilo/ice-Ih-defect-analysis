import argparse

def_keyfile = "./data-cache/CL-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
import numpy as np
import MDAnalysis as mda
from icedfmods.Structure_identification import get_single_rdf
from MDAnalysis import transformations as trans
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

RMIN = 0.5
RMAX = 6.5
NBINS = 200

# Received from main: data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all
def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, T, runparams, is_run_all, rmin, rmax, nbins):

    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    traj_file = dir_in / f'traj-{dir_in.name}.dcd'

    rdf_save = out_dir_rich / f'{dir_in.name}-rdf.npz'

    # If in update mode and all output files found, skip it
    if not(is_run_all) and rdf_save.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # Set up simulation
    u = mda.Universe(pdbin.absolute(), traj_file.absolute(), format = 'dcd')
    u.dimensions = cell_dims

    # Set up rdf dict
    RDFTYPES = ['OO', 'HH', 'OH']
    rdf_dict =  {rdf_type : np.zeros(nbins) for rdf_type in RDFTYPES}
    rdf_counts = {rdf_type : 0 for rdf_type in RDFTYPES}
    rdf_props = {'r_vals' : None, 'rmax' : rmax, 'nbins' : nbins, 'rdftypes' : RDFTYPES}
    ref_atoms = {name : u.select_atoms(f'name {name}') for name in ['O', 'H']}
    r_vals = np.zeros(nbins, dtype = int)

    # Get rdfs
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')

    # Iterate through rdf types that may be of interest
    for rdft in RDFTYPES:
        for ti in range(len(u.trajectory)):
            u.trajectory[ti]
            rdf, r_vals = get_single_rdf(u, ref_atoms[rdft[0]].indices, ref_atoms[rdft[1]], rmin, rmax, nbins)
            rdf_dict[rdft] += rdf
            rdf_counts[rdft] += 1

    rdf_props['r_vals'] = r_vals

    # Average out all the observed rdfs
    for key in rdf_dict.keys():
        if rdf_counts[key] > 0: rdf_dict[key] /= rdf_counts[key]

    # Save output
    rdf_savedict = rdf_dict | rdf_props
    np.savez_compressed(rdf_save, **rdf_savedict)

    # Result is the diagnostic for printing from the executor
    return f'{dft}-{run_num:02d}/{dir_in.name} completed'

def main():

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

    # Use only clean systems
    if "p0m0" not in dftypes: 
        raise(Exception(f"Clean p0m0 system not found in {dftypes}"))
    else:
        dftypes = ["p0m0"]

    run_idxs = runparams["run_indices"]

    # Simulation variables: not the main thing we want to converge, but different cases of it to compare over
    temps = runparams["temperature"]

    # Get input files
    pdbin_dir = Path(runparams["pdb_input_dir"])
    data_dir = Path(runparams["input_dir"])

    # Get output directory
    out_dir = Path("./data-cache") / runparams["parent_folder"]
    if not(out_dir.exists()): out_dir.mkdir()

    # Prepare the inputs for each run

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:
            # Get pdb input (determined by pXmY-ZZ)
            pdbname = f"{dft}-{run_num:02d}.pdb"
            pdbin = pdbin_dir / pdbname
            ref_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims

            # Make parent directory (pxmY-ZZ)
            out_dir_rich = out_dir / f"{dft}-{run_num:02d}"
            if not(out_dir_rich.exists()): out_dir_rich.mkdir()

            # Continue iterating over dependents
            for T in temps:
                # Save inputs for each given task
                task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all, RMIN, RMAX, NBINS))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(temps)])
    max_workers = np.min([tot_workers, 28])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())

if __name__ == '__main__':
    main()