"""
This script will find the total dipole moment for a water system.
What I mean by this is simply the position vector of each particle
from the center of the box, multiplied by its partial charge, for 
each timestep in a given simulation.
"""

import argparse
from icedfmods.Helper_modules import DATA_CACHE

def_keyfile = DATA_CACHE / "templates/CL-production.json"

parser = argparse.ArgumentParser()
parser.add_argument("from_file", help = "Key file to read data from", default = def_keyfile, type = str)
parser.add_argument("-a", "--all", help = "Whether to overwrite existing files", action="store_true")

args = parser.parse_args()

import numpy as np
import MDAnalysis as mda
from pathlib import Path
import re
from icedfmods.Helper_modules import submit_parallel_processes
from icedfmods.Structure_identification import get_total_dipole

# Partial charges (in e) taken from TIP4P/Ice (https://docs.lammps.org/Howto_tip4p.html)
PARTIAL_O = -1.1794 
PARTIAL_H = 0.5897

# Received from main: data_dir, pdbin, out_dir_rich, dft, run_num, ref_dims, T, runparams, run_all
def run_single_file(data_dir, pdbin, out_dir_rich, dft, run_num, cell_dims, T, runparams, is_run_all):

    inputmap = [dft, f"{run_num:02d}", T]
    input_formatted = re.sub(r'XXX[^X]*XXX', '{}', runparams["input_fmt"]).format(*inputmap)
    dir_in = data_dir / input_formatted

    traj_file = dir_in / f'traj-{dir_in.name}.dcd'
    dipole_save = out_dir_rich / f'{dir_in.name}-dipole.npy'

    # If in update mode and all output files found, skip it
    if not(is_run_all) and dipole_save.exists():
        return f'{dft}-{run_num:02d}/{dir_in.name} skipped'

    u = mda.Universe(pdbin, traj_file)
    u.dimensions = cell_dims

    dipole = get_total_dipole(u, PARTIAL_O, PARTIAL_H)

    np.save(dipole_save, dipole)

    # Result is the diagnostic for printing from the executor
    return f'{dft}-{run_num:02d}/{dir_in.name} completed'


if __name__ == '__main__':
    submit_parallel_processes(args, DATA_CACHE, run_single_file)
