import numpy as np
from MDAnalysis.analysis import distances as mddist

def get_frame_dependent_rdf(u, focus_idxs, reference_group, rmax, nbins):

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
        in_range = dists[dists < rmax]

        # Cast data to individual bins
        bins = (in_range / rmax * nbins).astype(int)

        # Count occurences of each value (cheaper np.histogram)
        counts += np.bincount(bins, minlength=nbins)    

    # Center histogram bins
    r_vals = (edges[1:] + edges[:-1]) / 2

    # Normalization stuff
    shell_volume = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    pair_density = N_pairs / np.prod(dims[:3]) # Accounts for all timesteps in N_pairs

    # Get output rdf
    rdf = counts / (shell_volume * pair_density)

    return rdf, r_vals