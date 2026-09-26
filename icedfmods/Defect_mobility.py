import numpy as np
import MDAnalysis as mda
from tqdm import tqdm

def get_lifetime_distribution(df_dict, name):

    # Get mask identifying defect type
    DFTYPES = df_dict["type_names"]
    type_code = list(DFTYPES).index(name)
    name_mask = df_dict["type"] == type_code

    if not name_mask.any():
        return None

    # Get list of all atom idxs associated with that type
    all_atoms = np.vstack([df_dict["atom1"][name_mask], df_dict["atom2"][name_mask]])
    all_frame = df_dict["frame"][name_mask]
    all_track = df_dict["track_id"][name_mask]

    # Order by track, frame, atom1, atom2
    order = np.lexsort([all_atoms[1, :], all_atoms[0, :], all_frame, all_track])

    track_s = all_track[order]
    atoms_s = all_atoms[:, order]
    frame_s = all_frame[order]

    track_changed = track_s[1:] != track_s[:-1]
    if name == "L":
        # Resolve atom1, atom2 to make the top row atom smaller by construction
        atoms_L = np.sort(atoms_s, axis = 0)
        site_changed = (atoms_L[:, 1:] != atoms_L[:, :-1]).any(axis=0)
    else:
        site_changed = atoms_s[0, 1:] != atoms_s[0, :-1]

    changed = site_changed + track_changed

    starts = np.concatenate([[True], changed])
    ends = np.concatenate([changed, [True]])

    lifetimes = frame_s[ends] - frame_s[starts] + 1 # 1 frame of life is 1 instead of 0

    return lifetimes


