import numpy as np
from MDAnalysis.analysis import distances as mddist

def get_frame_dependent_rdf(u, focus_idxs, reference_group, rmin, rmax, nbins):

    Nt = len(u.trajectory)
    dims = u.dimensions

    ref_idxs = reference_group.indices

    # Set up histogram properties
    edges = np.linspace(0, rmax, nbins + 1)
    counts = np.zeros(nbins, dtype = int)

    # Counter for # of atoms seen
    N_pairs = 0

    for ti in range(Nt):
        u.trajectory[ti]

        idx = focus_idxs[ti]

        # Filter out invalid options
        idx = idx[idx >= 0]
        if len(idx) == 0:
            continue

        focus_atom = u.atoms[idx]

        dists = mddist.distance_array(focus_atom.positions, reference_group.positions, box = dims)
        keep = ref_idxs[None, :] != idx[:, None]    # Filter out self-interaction
        dists = dists[keep]                         # Will flatten the array

        # Increment by number of possible pairs
        N_pairs += dists.size

        # Consider only pairs in range
        in_range = (dists < rmax) * (dists > rmin)

        # Count occurences of each value
        hist, _ = np.histogram(dists[in_range], bins = edges)
        counts += hist

    # Center histogram bins
    r_vals = (edges[1:] + edges[:-1]) / 2

    # Normalization stuff
    shell_volume = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    pair_density = N_pairs / np.prod(dims[:3]) # Accounts for all timesteps in N_pairs

    # Get output rdf
    rdf = counts / (shell_volume * pair_density)

    return rdf, r_vals

def get_single_rdf(u, focus_idxs, reference_group, rmin, rmax, nbins):

    dims = u.dimensions
    ref_idxs = reference_group.indices

    # Set up histogram properties
    edges = np.linspace(0, rmax, nbins + 1)
    counts = np.zeros(nbins, dtype = int)

    # Filter out invalid options
    idx = focus_idxs[focus_idxs >= 0]
    if len(idx) == 0:
        return None

    focus_atom = u.atoms[idx]

    dists = mddist.distance_array(focus_atom.positions, reference_group.positions, box = dims)
    keep = ref_idxs[None, :] != idx[:, None]    # Filter out self-interaction
    dists = dists[keep]                         # Will flatten the array

    # Save number of possible pairs
    N_pairs = dists.size

    # Consider only pairs in range
    in_range = (dists < rmax) * (dists > rmin)

    # Count occurences of each value
    hist, _ = np.histogram(dists[in_range], bins = edges)
    counts += hist

    # Center histogram bins
    r_vals = (edges[1:] + edges[:-1]) / 2

    # Normalization stuff
    shell_volume = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    pair_density = N_pairs / np.prod(dims[:3]) # Accounts for all timesteps in N_pairs

    # Get output rdf
    rdf = counts / (shell_volume * pair_density)

    return rdf, r_vals