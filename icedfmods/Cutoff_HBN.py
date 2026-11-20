"""
Module to count hydrogen bonds by the 
Luzar-Chandler definition (O...O dist < 3.5 A,
and O-H...O angle > 150deg)
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA


def build_hbonds(u : mda.Universe, blocks : int = 20):
    hbonds = HBA(
        universe = u,
        donors_sel = "name O",
        hydrogens_sel = "name H",
        acceptors_sel = "name O",
        d_h_cutoff = 1.5,           # Cutoff for donor/hydrogen pairs
        d_a_cutoff = 3.5,           # Cutoff for donor/acceptor OO distance
        d_h_a_angle_cutoff = 150    # Cutoff for donor-hydrogen...acceptor angle (degrees)
    )

    # Initialize output dict
    DICT_KEYS = ["frame", "donor", "hyd", "accep", "dist", "angle"]
    DATA_TYPES = [int, int, int, int, float, float]
    results_dict = {key : [] for key in DICT_KEYS}


    Nt = len(u.trajectory)
    edges = np.linspace(0, Nt, blocks + 1, dtype = int)
    for i in range(blocks):
        hbonds.run(start=edges[i], stop=edges[i+1])

        results = hbonds.results.hbonds

        for ik, (key, dtype) in enumerate(zip(DICT_KEYS, DATA_TYPES)):
            results_dict[key].append(results[:, ik].astype(dtype))

    # Concatenate each block to a giant 1D array
    return {k: np.concatenate(v) for k, v in results_dict.items()}

def defect_hbond_coordination(u: mda.Universe, hbonds : dict, hydidx, dfts):
    """
    Count the coordination number of the ionic defects which share
    a specific defect hydrogen hydidx (should work for both OH and H3O)
    """

    coord = np.full((2, len(dfts)), -1, dtype = int)
    for ti, ts in enumerate(dfts):
        u.trajectory[ts]

        mask_by_frame = hbonds["frame"] == ts
        mask_by_defect = (hbonds["hyd"] == hydidx[ti]) & mask_by_frame

        donor = hbonds["donor"][mask_by_defect]
        accep = hbonds["accep"][mask_by_defect]

        mask_by_donor = ((hbonds["donor"] == donor) | (hbonds["accep"] == donor)) & mask_by_frame
        mask_by_accep = ((hbonds["donor"] == accep) | (hbonds["accep"] == accep)) & mask_by_frame

        # Here, (coord[0], coord[1]) is Laage's (na, nb)
        coord[0, ti] = np.sum(mask_by_donor)
        coord[1, ti] = np.sum(mask_by_accep)

    return coord

def find_O4(u, Oc, partner, Hstar, box):
    """
    Fourth H-bond donor (O4, H4) around the central oxygen Oc (= Oa or Ob).
    Hstar   : index of the transferred proton
    partner : the other oxygen of the pair (Ob if Oc = Oa, and vice versa)
    Returns universe atom indices (O4, H4), or (-1, -1) if nothing is found.
    """
    oxy = u.select_atoms("name O")
    hyd = u.select_atoms("name H")
    rc = u.atoms[Oc].position
 
    dO = oxy.positions - rc
    dO -= box * np.rint(dO/box)
    rO = np.linalg.norm(dO, axis = 1)
    
    dH = hyd.positions - rc
    dH -= box * np.rint(dH/box)
    rH = np.linalg.norm(dH, axis = 1)
 
    def closest_H_on(Oj):
        # H covalently bonded to Oj (same 1.2 A cutoff as HBA) that is closest to Oc
        dHO = hyd.positions - u.atoms[Oj].position
        dHO -= box * np.rint(dHO/box)
        bonded = np.linalg.norm(dHO, axis = 1) < 1.2
        if not bonded.any():
            return -1, np.inf
        k = np.argmin(np.where(bonded, rH, np.inf))
        return hyd.indices[k], rH[k]

    def closest_O_on(Hj, exclude = ()):
        # O which accepts Hj that is not exclude
        dOH = oxy.positions - u.atoms[Hj].position
        dOH -= box * np.rint(dOH/box)
        dOH = np.linalg.norm(dOH, axis = -1)
        k = np.argmin(np.where(np.isin(oxy.indices, exclude), np.inf, dOH))
        return oxy.indices[k]
 
    # Step 1: does the 4th-closest oxygen donate an H within 2.5 A of Oc?
    rank = np.where(oxy.indices == Oc, np.inf, rO)
    O4 = oxy.indices[np.argsort(rank)[3]]
    H4, dist = closest_H_on(O4)
    if dist < 2.5 and O4 != partner:
        O4a = closest_O_on(H4, exclude=[O4, Oc])
        return O4, H4, O4a
 
    # Step 2: cone. Three H = H* + the two H closest to Oc other than H*
    #         (vectors relative to Oc, so Oc sits at the origin)
    own = np.argsort(np.where(hyd.indices == Hstar, np.inf, rH))[:2]    # 2 closest non H* hydrogens
    h = np.vstack([dH[hyd.indices == Hstar][0], dH[own]])               # 3 hydrogens- H* and two closest H to Oc
 
    n = np.cross(h[1] - h[0], h[2] - h[0])      # Cone axis unit vector
    n /= np.linalg.norm(n)
    if np.dot(n, -h.mean(axis = 0)) < 0:     # point from the H plane through Oc, outward
        n = -n
 
    cos = (dO @ n) / np.where(rO > 0, rO, np.inf)
    in_cone = (cos >= np.cos(np.radians(60))) & ~np.isin(oxy.indices, [Oc, partner])
    if not in_cone.any():
        return -1, -1, -1
 
    O4 = oxy.indices[in_cone][np.argmin(rO[in_cone])]
    H4, _ = closest_H_on(O4)

    if H4 < 0:
        return -1, -1, -1

    O4a = closest_O_on(H4, exclude=[O4, Oc])
    return O4, H4, O4a

def get_Laage_coordinate(u, hbonds, hydidx, dfts):
    """
    Get defect second-shell coordination as in Laage et. al's
    '... doubly gated by sequential hydrogen-bond exchange'

    Only defined for H3O+
    """
    box = u.dimensions[:3]
    O4 = np.full((2, len(dfts)), -1, dtype = int)
    H4 = np.full((2, len(dfts)), -1, dtype = int)
    O4acc = np.full((2, len(dfts)), -1, dtype = int)
    gamma = np.full(len(dfts), np.nan)

    # Get frame slices first to make things faster
    frame_slices = slice_hbond_frames(hbonds["frame"])

    for ti, ts in enumerate(dfts):
        sl = frame_slices.get(ts)
        if sl is None:
            continue

        don, acc, hyd = hbonds["donor"][sl], hbonds["accep"][sl], hbonds["hyd"][sl]
        mask_by_defect = hyd == hydidx[ti]

        # Increase u timestep
        u.trajectory[ts]

        if np.count_nonzero(mask_by_defect) < 1:
            continue

        elif np.count_nonzero(mask_by_defect) == 1:
            # Here, Ob is the one accepting the defect hydrogen which Oa donates
            Oa = don[mask_by_defect][0]         # Unpack to scalars
            Ob = acc[mask_by_defect][0]

        else:
            cand = don[mask_by_defect]
            dist = u.atoms[cand].positions - u.atoms[hydidx[ti]].position
            dist -= box * np.rint(dist/box)

            winner = np.argmin(np.linalg.norm(dist, axis = -1))
            Oa = cand[winner]
            Ob = acc[mask_by_defect][winner]

        # Get H4 and O4 for each Oa, Ob
        O4[0, ti], H4[0, ti], O4acc[0, ti] = find_O4(u, Oa, Ob, hydidx[ti], box)
        O4[1, ti], H4[1, ti], O4acc[1, ti] = find_O4(u, Ob, Oa, hydidx[ti], box)

        # Skip out on crash
        if (H4[:, ti] < 0).any():
            continue

        # Find actual coordinate
        H4_pos = u.atoms[H4[:, ti]].positions
        O4acc_pos = u.atoms[O4acc[:, ti]].positions
        delta_a = np.linalg.norm(pbc_dist(H4_pos[0], u.atoms[Oa].position, box), axis = -1) - np.linalg.norm(pbc_dist(H4_pos[0], O4acc_pos[0], box), axis = -1)
        delta_b = np.linalg.norm(pbc_dist(H4_pos[1], u.atoms[Ob].position, box), axis = -1) - np.linalg.norm(pbc_dist(H4_pos[1], O4acc_pos[1], box), axis = -1)

        gamma[ti] = np.arctan(delta_b / delta_a) + np.pi * (1 - (delta_a > 0))

    return gamma, (O4, H4, O4acc)

def get_delta_coordinate(u: mda.Universe, hbonds : dict, dfidx, dfts):
    """
    Compute ionic delta coordinate
    """

    box = u.dimensions[:3]

    coords = np.full(len(dfts), np.nan)
    sums = np.full(len(dfts), np.nan)
    hydidxs = np.full(len(dfts), -1, dtype = int)

    # Get frame slices first to make things faster
    frame_slices = slice_hbond_frames(hbonds["frame"])

    for ti, ts in enumerate(dfts):

        sl = frame_slices.get(ts)
        if sl is None:
            continue

        don, acc, hyd = hbonds["donor"][sl], hbonds["accep"][sl], hbonds["hyd"][sl]
        mask_by_defect = (don == dfidx[ti]) | (acc == dfidx[ti])

        if not(mask_by_defect.any()):
            continue

        u.trajectory[ts]

        # For each candidate hydrogen bond, compute delta
        hyds = hyd[mask_by_defect]
        donors = don[mask_by_defect]
        acceps = acc[mask_by_defect]

        # Get defect oxygen
        dfO = dfidx[ti]
        pairs = np.where(donors == dfO, acceps, donors) # Where the donor is dfO, return from accep
        
        # Get positions
        dfO_pos = u.atoms[dfO].position
        H_pos = u.atoms[hyds].positions
        ptO_pos = u.atoms[pairs].positions

        # Get distances (PBC-corrected)
        df_h = np.linalg.norm(pbc_dist(H_pos, dfO_pos, box), axis = -1)
        pt_h = np.linalg.norm(pbc_dist(H_pos, ptO_pos, box), axis = -1)

        # Compute all deltas, then choose the one closest to 0
        delta_candidates = df_h - pt_h
        best = np.argmin(np.abs(delta_candidates))

        coords[ti] = delta_candidates[best]
        hydidxs[ti] = hyds[best]

        # Get summed distance
        df_pt = np.linalg.norm(pbc_dist(dfO_pos, ptO_pos[best], box))
        sums[ti] = df_pt

    return coords, sums, hydidxs

def slice_hbond_frames(frames):
    """
    Slice an increasing array of frames in Slice objects so I don't have to do huge boolean masks all the time
    """

    assert np.all(frames[1:] >= frames[:-1]), "hbonds not frame-sorted"
    starts = np.flatnonzero(np.r_[True, frames[1:] != frames[:-1]]) # np.r_ is like np.concatenate here
    ends = np.r_[starts[1:], len(frames)]
    frame_slices = {frames[st] : slice(st, en) for st, en in zip(starts, ends)}

    return frame_slices

def pbc_dist(a, b, box):
    d = a - b
    d -= box * np.rint(d/box)
    return d