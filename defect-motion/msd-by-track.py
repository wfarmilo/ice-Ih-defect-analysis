import argparse
from icedfmods.Helper_modules import DATA_CACHE

def_keyfile = DATA_CACHE / "templates/CL-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action = "store_true", help = "Run for all systems")
parser.add_argument("-ff", "--from_file", help = "Key file to read data from", default = def_keyfile, type = str)

args = parser.parse_args()

import json
from icedfmods.MSD_FFT import get_fft_msd, fill_frame_positions
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Which defects to use midpoint position for
MIDPOINT_DEFECT = ["L"]

# Received from main: data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all
def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, T, runparams, is_run_all, LIFETIME, RMAX):

    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    traj_file = dir_in / f'traj-{dir_in.name}.dcd'
    df_path = out_dir_rich / f'{dir_in.name}-defects.npz'
    msd_savepath = out_dir_rich / f'{dir_in.name}-msd.npz'

    # If in update mode and all output files found, skip it
    if not(is_run_all) and msd_savepath.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    # Set up simulation
    u = mda.Universe(pdbin.absolute(), traj_file.absolute(), format = 'dcd')
    u.dimensions = cell_dims

    # Get COM
    COM = np.zeros((len(u.trajectory), 3))
    for ti, ts in enumerate(u.trajectory):
        COM[ti, :] = u.atoms.center_of_mass()

    # Get all defect types
    df_dict = np.load(df_path)
    DFTYPES = df_dict["type_names"]
    tids = {name : [] for name in DFTYPES}

    # Initialize save list
    msdsave = []

    # Find all msd's
    print(f'Starting {dft}-{run_num:02d}/{dir_in.name}')

    for type_code, name in enumerate(DFTYPES):
        mask_by_name = df_dict["type"] == type_code
        tids[name] = np.unique(df_dict["track_id"][mask_by_name])
        for tid in tids[name]:
            mask_by_tid = (df_dict["track_id"] == tid) * mask_by_name

            # Build continuous trajectory out of frames
            frames, positions = fill_frame_positions(u, name, df_dict["frame"][mask_by_tid], df_dict["atom1"][mask_by_tid], df_dict["atom2"][mask_by_tid], MIDPOINT_DEFECT)
            pos_shifted = positions - COM[frames, :]

            # Skip empty results (single frame defects)
            if pos_shifted.shape[0] == 0:
                continue

            msd = get_fft_msd(pos_shifted, unwrapped = False, box = u.dimensions[:3])

            # Add total x,y,z contributions to msd save
            msd3d = np.sum(msd, axis = 1)

            # Build huge list of individual values to compress to .npz
            t0 = frames[0]
            for tau, msdval in enumerate(msd3d):
                msdsave.append((tid, type_code, t0, tau, msdval))

    # Refactor msd_dict- now contains each tid within the dftype, as well as a list of all tids under ["OH"]["track_ids"]
    track_ids, type_codes, t0s, taus, msdval3d = zip(*msdsave) if msdsave else ([], [], [], [], [])

    # Build dict
    msd_dict = {
        "track_id" : np.array(track_ids, dtype = int),
        "type" : np.array(type_codes, dtype = int),
        "t0" : np.array(t0s, dtype = int),
        "tau" : np.array(taus, dtype = int),
        "msd" : np.array(msdval3d, dtype = float),
        "type_names" : DFTYPES
    }
    

    # Save output
    np.savez_compressed(msd_savepath, **msd_dict)

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
    run_idxs = runparams["run_indices"]

    # Use only defect systems
    if "p0m0" in dftypes: dftypes.remove("p0m0")

    # Simulation variables: not the main thing we want to converge, but different cases of it to compare over
    temps = runparams["temperature"]

    # Get input files
    pdbin_dir = Path(runparams["pdb_input_dir"])
    data_dir = Path(runparams["input_dir"])

    pdbin_fmt = runparams["pdb_fmt"]

    # Get output directory
    out_dir = DATA_CACHE / runparams["parent_folder"]

    # Prepare the inputs for each run

    # Defect matching parameters
    LIFETIME = runparams["defect_lifetime"]
    RMAX = runparams["defect_maxjump"]

    # List of inputs to out function to run
    task_params = []

    for dft in dftypes:
        for run_num in run_idxs:
            # Get pdb input (determined by pXmY-ZZ)
            pdbname = re.sub(r'XXX[^X]*XXX', '{}', pdbin_fmt).format(dft, f"{run_num:02d}")
            pdbin = pdbin_dir / pdbname
            ref_dims = mda.Universe(pdbin.absolute()).dimensions # Reference Universe for cell dims

            # Make parent directory (pxmY-ZZ)
            out_dir_rich = out_dir / f"{dft}-{run_num:02d}"

            # Continue iterating over dependents
            for T in temps:
                # Save inputs for each given task
                task_params.append((data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all, LIFETIME, RMAX))

    # Queue all processes
    tot_workers = np.prod([len(dftypes), len(run_idxs), len(temps)])
    max_workers = np.min([tot_workers, 28])

    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        task_returns = [executor.submit(run_single_file, *tp) for tp in task_params]
        for ret in as_completed(task_returns):
            print(ret.result())

if __name__ == '__main__':
    main()