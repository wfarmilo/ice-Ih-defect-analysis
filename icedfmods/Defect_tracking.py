import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances as mddist
from tqdm import tqdm
from freud import box as fdbox, locality as fdloc
from scipy.optimize import linear_sum_assignment

def raw_oxyNeighbourList(oxy, box, cutoff = 0):
    No = len(oxy)
    fbox = fdbox.Box(*box) #Assumes orthogonal box
    points = oxy.positions

    voro = fdloc.Voronoi()

    voro.compute((fbox, points))

    nlist = voro.nlist

    i = nlist.query_point_indices
    j = nlist.point_indices
    d = nlist.distances
    w = nlist.weights

    # Make connection matrix
    pairs = np.zeros((No, No), dtype = float)
    pairs[i, j] = w

    # Get indices for upper triangular part of pairs
    iu, ju = np.triu_indices(No, k=1)

    # Keep only pairs that are actually neighbours with w > cutoff (by default is just 0)
    valid = pairs[iu, ju] > cutoff
    iu_valid = iu[valid]
    ju_valid = ju[valid]

    # Build ragged neighbourlist
    max_nn = 0
    NNlist = [[] for _ in range(No)]
    weights = [[] for _ in range(No)]
    for Oi, Oj in zip(iu_valid, ju_valid):
        NNlist[Oi].append(Oj)
        NNlist[Oj].append(Oi)
        max_nn = np.max([max_nn, len(NNlist[Oi]), len(NNlist[Oj])])

        # Add weights
        weights[Oi].append(pairs[Oi, Oj])
        weights[Oj].append(pairs[Oj, Oi])   # Same value

    for row, wrow in zip(NNlist, weights):
        row.extend((max_nn - len(row)) * [-1])
        wrow.extend((max_nn - len(wrow)) * [0])

    NNlist = np.array(NNlist).reshape((No, max_nn))
    weights = np.array(weights).reshape((No, max_nn))

    return NNlist, weights

def get_hydNeighborList(oxy, hyd, dim, cutoff = 3.0):

    pairs, dists = mddist.capped_distance(hyd, oxy, max_cutoff = cutoff, box = dim)

    #Sorts first by hydrogen index, then by distance, so our output looks like:
    # h_ind:  0    0    0    1    1    2  etc.
    #  dist: 0.1  1.2  2.9  1.1  1.2  0.5
    order = np.lexsort((dists, pairs[:, 0]))
    pairs_s = pairs[order, :]

    #Find the indices of the first occurence of each hydrogen
    unique, indices = np.unique(pairs_s[:, 0], return_index = True)

    HO_pairs = pairs_s[indices, 1]

    return HO_pairs

def get_hbond_neighbours(u, oxyNL_ragged, weights):
    """Creates directed graph out of a given ice universe"""

    oxy = u.select_atoms("name O")
    hyd = u.select_atoms("name H")

    #Useful quantities
    box = u.dimensions[:3]
    No = len(oxy)
    Nh = len(hyd)

    #Get hydrogen configurations
    hydoxyNL = get_hydNeighborList(oxy, hyd, u.dimensions, cutoff = 3.0)

    #Make sure we didn't lose anybody from the cutoff fxn
    assert len(hydoxyNL) == Nh

    #Get all OH vectors
    OH_pos = oxy[hydoxyNL].positions #The position of the oxygen attached to each hydrogen
    OH_vecs = hyd.positions - OH_pos
    OH_vecs -= box[None, :] * np.rint(OH_vecs/box[None, :])

    #O neighbours for each hydrogen
    NN_O_idx = oxyNL_ragged[hydoxyNL, :]   #shape (Nh, Nn)
    NN_O_pos = oxy.positions[NN_O_idx] #shape will be (Nh, Nn, 3) for [Hk, Oj, xyz]

    #For each neighboring Oj, find the OO distance between it and our hydrogen's oxygen Oi
    OO_vecs = NN_O_pos - OH_pos[:, None, :]
    OO_vecs -= box[None, None, :] * np.rint(OO_vecs/box[None, None, :])

    #Normalize
    OH_vecs /= np.linalg.norm(OH_vecs, axis = -1)[:, None]
    OO_vecs /= np.linalg.norm(OO_vecs, axis = -1)[:, :, None]

    #Compute dot product
    #Here, 'ij,ikj->ik' translates as follows:
    #   i: hydrogen index
    #   j: dimension index (xyz)
    #   k: oxygen neighbour index (1..4)
    # Then, what this sum is doing is for each for each element ij in OH_vecs, it 
    # adds the element ikj in OO_vecs and stores it in element ik of prod. Thus,
    # we sum over all of the xyz coordinates j of the product (denoted by ',' here)
    prod = np.einsum('ij,ikj->ik', OH_vecs, OO_vecs)    #Shape (Nh, Nn)

    # Use combined score from voronoi neighbour weighting and angle in order
    # to filter out really far neighbours
    NN_weights = weights[hydoxyNL, :]
    valid = NN_weights > 0

    # Take product at valid sites (mask out invalid ones with impossibly low score)
    score = np.where(valid, prod * np.sqrt(NN_weights), -np.inf)

    # Find best score for each hydrogen
    sorted_idx = np.argsort(score, axis = 1)

    # Pick out highest score to pair with
    best_O = sorted_idx[:, -1]

    # Save angles for analysis
    best_angles = prod[np.arange(Nh), best_O]

    # Build acceptor and donor pairs
    accep_O = NN_O_idx[np.arange(Nh), best_O]   #Shape (Nh,)
    donor_O = hydoxyNL                          #Shape (Nh,)

    return donor_O, accep_O, best_angles

def classify_defects(donor_O, accep_O, oxyNL_ragged, bestangles, oxy2uni):
    """
    Returns oxygen indices of classified defects as a dict by defect name
    """
    No = len(oxyNL_ragged)
    oind = np.arange(No)

    # Counts how many hydrogens each oxygen is donating
    donor_cts = np.bincount(donor_O, minlength = No)

    # Use coordination for ionic
    OH_idx = oind[donor_cts == 1]
    H3O_idx = oind[donor_cts == 3]

    # Stack ionic defects with empty flags to preserve shape
    OH_idx = np.vstack([OH_idx, np.full(OH_idx.shape, -1, dtype = int)]).T
    H3O_idx = np.vstack([H3O_idx, np.full(H3O_idx.shape, -1, dtype = int)]).T

    # For Bjerrum, should correspond to incorrect total counds, resolved by reconstructing hbond network
    L_pairs, D_pairs = find_network_defects(No, donor_O, accep_O, oxyNL_ragged, bestangles)

    # Return dict of defects keyed by name
    classified = {
        'OH': oxy2uni[OH_idx],
        'H3O': oxy2uni[H3O_idx],
        'L': oxy2uni[L_pairs],
        'D': oxy2uni[D_pairs]
    }

    return classified

def find_network_defects(No, donor_O, accep_O, oxyNL_ragged, bestangles):

    # Build oxyNL from hbonds
    edge_da = np.concatenate([donor_O, accep_O])
    edge_ad = np.concatenate([accep_O, donor_O])
    order = np.argsort(edge_da)

    # Sort da array first, then match the other atom each is paired with
    # Example: Given (4,2), (1,3), (2,1), (1,2)
    # da_sorted: 1 1 2 4
    # ad_sorted: 3 2 2 4 (not necessarily in increasing order!)
    da_sorted, ad_sorted = edge_da[order], edge_ad[order]
    _, bounds = np.unique(da_sorted, return_index = True)   # First occurence of each oxygen in da
    bounds = np.concatenate([bounds, [len(da_sorted)]])   # Pad in last element for entry No

    oxyNL_bondcounts = bounds[1:] - bounds[:-1]     # Should have length No
    oxyNL_bonds = [ad_sorted[bounds[o]:bounds[o + 1]] for o in range(No)]

    # Identify defect rows
    L_rows = np.arange(No)[oxyNL_bondcounts < 4]
    D_rows = np.arange(No)[oxyNL_bondcounts > 4]

    # D defect: Find the doubled bond in oxyNL_bonds
    duped_symmetric_keys = []
    for dr in D_rows:
        seen = set()
        duplicates = list({i for i in oxyNL_bonds[dr] if i in seen or seen.add(i)})   # Either add to list or add to seen
        keys = [min(dr, dup) * No + max(dr, dup) for dup in duplicates]

        duped_symmetric_keys.extend(keys)

    # Then, we pair D defects by checking which symmetric keys are duplicated
    seen = set()
    dupes = np.array(list({i for i in duped_symmetric_keys if i in seen or seen.add(i)}))

    D_pairs = []

    # Sort by worst angle then best angle (for atomwise def'n)
    for dup in dupes:
        pair = np.array([dup // No, dup % No], dtype = int)

        # Guaranteed sorted by lowest index -> highest index by construction
        first = (donor_O == pair[0]) * (accep_O == pair[1])
        second = (donor_O == pair[1]) * (accep_O == pair[0])

        # Skip pairs who double-donate or double-receive
        if not(first.any()) or not(second.any()):
            continue

        # Find angles for both cases
        angs = [bestangles[first][0], bestangles[second][0]]

        #Smallest angs value is dangling H-bond
        sortinds = np.argsort(angs)

        # Assign to D_pairs
        D_pairs.append(pair[sortinds])

    # Turn into array
    D_pairs = np.array(D_pairs, dtype = int)

    # L defect: find which oxygen is missing in oxyNL_bonds from both sides
    duped_symmetric_keys = []
    for lr in L_rows:

        # Neighbouring oxygens who weren't connected by hbond network are up for grabs
        candidates = set(oxyNL_ragged[lr]) - set(oxyNL_bonds[lr])

        # Make symmetric keys between lr and each candidate it could bond to
        keys = [min(lr, cd) * No + max(lr, cd) for cd in candidates]
        duped_symmetric_keys.extend(keys)

    # Look for duplicate keys (matched pairs)
    seen = set()
    dupes = np.array(list({i for i in duped_symmetric_keys if i in seen or seen.add(i)}))

    L_pairs = np.empty((len(dupes), 2), dtype = int)
    L_pairs[:, 0] = (dupes // No).astype(int)
    L_pairs[:, 1] = dupes % No

    return L_pairs, D_pairs

def build_validjump_single(u_frame, df, RMAX):
    """
    Find the valid atoms which could conceivably have jumped to the defect position
    returns in the same shape as df arraw input with extra N_neighbours dimension, computed row-wise
    df is of shape (N_defects, N_atoms/defect)
    """

    oxy = u_frame.select_atoms("name O")
    save = [[] for _ in range(df.shape[1])]
    max_len = 1

    # Loop over all atom pairs and stack them- here "row" is a given atom coord
    for ri, row in enumerate(df.T):
        row_mask = row >= 0

        # Skip if empty (should be true for ionic defects)
        if np.count_nonzero(row_mask) == 0:
            continue

        pairs, dists = mddist.capped_distance(oxy[row[row_mask]], oxy, min_cutoff=0.01, max_cutoff=RMAX, box=u_frame.dimensions, return_distances=True)
        unique, counts = np.unique(pairs[:, 0], return_counts = True)

        order = np.lexsort([dists, pairs[:, 0]])
        pairs_s = pairs[order, :]

        ischanged = np.concatenate(([True], pairs_s[1:, 0] != pairs_s[:-1, 0], [True]))
        changed_ind = np.nonzero(ischanged)[0]
        spacing = changed_ind[1:] - changed_ind[:-1]

        counter = np.concatenate([np.arange(cgd) for cgd in spacing])

        max_len = max(np.max(counts), max_len)

        save[ri] = [pairs, counter, dists]


    out = np.full((df.shape[0], df.shape[1], max_len), -1, dtype = int)
    out_dists = np.full((df.shape[0], df.shape[1], max_len), np.inf, dtype = float)

    for ri, row in enumerate(df.T):

        if not(save[ri]): continue

        pairs, counter, dists = save[ri]

        out[pairs[:, 0], ri, counter] = pairs[:, 1]
        out_dists[pairs[:, 0], ri, counter] = dists

    return out, out_dists

def match_defects(u, new_idxs, old_idxs, old_trackids, old_lifetimes, next_id, max_lifetime, uni2oxy, RMAX):

    N_new = len(new_idxs)
    N_old = len(old_idxs)

    #Output: new ids for each oxygen
    new_trackids = np.full(N_new, -1, dtype = int)

    #Array of valid candidates for each old atom (in oxygen indices)
    allowed_idx, allowed_dist = build_validjump_single(u, old_idxs, RMAX)  #Shape N_old, N_atoms, N_neighbours

    #Boolean of possible sources for each atom
    possible_sources = np.zeros((N_new, N_old), dtype = bool)
    distance_sources = np.full((N_new, N_old), np.inf, dtype = float)   # Large sentinel value

    #Indexes (in new/old_idxs)
    Old, New = np.meshgrid(np.arange(N_old), np.arange(N_new))

    #For each new candidate, check possible sources
    for ni in range(N_new):
        mask_idxs = new_idxs[ni, :] >= 0
        new_uni_idx = uni2oxy[new_idxs[ni, mask_idxs]]  #Shape N_atoms (2 or 1)
        inhop = (new_uni_idx[None, :, None] == allowed_idx) + (new_uni_idx[None, ::-1, None] == allowed_idx) #shape (N_old, N_atoms, N_neghbours)
        issource = inhop.any(axis=(1,2))        # Collapse to N_old checking if a given old atom is a source
        
        masked_dist = np.where(inhop, allowed_dist, np.inf)
        closest_dist = masked_dist.min(axis=(1, 2))                       # real minimum distance

        possible_sources[ni] = issource
        distance_sources[ni] = closest_dist

    #Locate singly sourced and singly received
    source_count = np.count_nonzero(possible_sources, axis = 1) #shape N_new, number of sources per new output
    output_count = np.count_nonzero(possible_sources, axis = 0) #shape N_old, number of outputs per old source

    #Reconstruct (N_new, N_old) shape, ensuring that we capture real pairs
    one_to_one = (source_count == 1)[:, None] * (output_count == 1)[None, :] * possible_sources   #Shape (N_new, N_old), identifies 1-1 pairs

    #Assign new track id to match old track id
    new_trackids[New[one_to_one]] = old_trackids[Old[one_to_one]]

    #Clear used pairs
    possible_sources[one_to_one] = False

    #Assign more complicated pairs
    resolved = [True]
    while np.count_nonzero(resolved) > 0:

        #Get source/receive counts
        source_count = np.count_nonzero(possible_sources, axis = 1) #shape N_new, number of sources per new output
        output_count = np.count_nonzero(possible_sources, axis = 0) #shape N_old, number of outputs per old source

        #Check for one-sided uniqueness (1 source or 1 output)
        oneside = ((source_count == 1)[:, None] + (output_count == 1)[None, :]) * possible_sources

        #Escape early to save compute
        if not oneside.any():
            break

        #Resolve shared output overlap
        closest_by_output = np.where(oneside, distance_sources, np.inf)         #masked array of shape (N_new, N_old) with distance values filled in where one-sided pairs exist
        winner_by_output = np.argmin(closest_by_output, axis = 0)               #Find closest paired output for each source (shape N_old)
        source_winner_ispaired = oneside.any(axis = 0)                          #Per source, is any output trying to pair with it

        #Construct matrix of winning sources for each conflicting output
        source_winners = np.zeros((N_new, N_old), dtype = bool)
        source_winners[winner_by_output[source_winner_ispaired], np.arange(N_old)[source_winner_ispaired]] = True   #Now each source has at most one output paired with it

        #Resolve shared source overlap
        closest_by_source = np.where(source_winners, distance_sources, np.inf)  #masked array of shape (N_new, N_old) with distance ranks filled in where we have source winners
        winner_by_source = np.argmin(closest_by_source, axis = 1)               #Find closest paired source for each output (shape N_new)
        output_winner_ispaired = source_winners.any(axis = 1)                   #Per output, is there a source trying to pair with it

        #Construct final matrix of winning outputs for each winning source
        resolved = np.zeros((N_new, N_old), dtype = bool)
        resolved[np.arange(N_new)[output_winner_ispaired], winner_by_source[output_winner_ispaired]] = True #Now, each output has one source paired, from the previously trimmed source_winners

        #Assign trackids
        new_trackids[New[resolved]] = old_trackids[Old[resolved]]

        #Remove rows/cols from possible_sources
        resolved_new = resolved.any(axis = 1)   #Claimed outputs
        resolved_old = resolved.any(axis = 0)   #Claimed sources

        possible_sources[resolved_new, :] = False   #No other sources can take this output
        possible_sources[:, resolved_old] = False   #No other outputs can take this source

    # Fallback: leftover ambiguity the count-based peel couldn't break (e.g. a symmetric tie) gets
    # resolved by nearest hop-distance
    remaining_new = np.nonzero(possible_sources.any(axis = 1))[0]
    remaining_old = np.nonzero(possible_sources.any(axis = 0))[0]
    if remaining_new.size and remaining_old.size:
        # Get cross product (all possible pairs) of remaining outputs
        sub_possible = possible_sources[np.ix_(remaining_new, remaining_old)]

        # Fill in distance based cost (sentinel np.inf)
        sub_cost = np.where(sub_possible, distance_sources[np.ix_(remaining_new, remaining_old)], np.inf)

        # Match by distance weighting for those who still need matching
        row_ind, col_ind = linear_sum_assignment(sub_cost)
        keep = sub_possible[row_ind, col_ind]
        new_trackids[remaining_new[row_ind[keep]]] = old_trackids[remaining_old[col_ind[keep]]]

    # Match unassigned defects with stale ones (one shot)
    already_matched = set(new_trackids.tolist())
    for tid, (idx, stale_count) in old_lifetimes.items():
        if tid in already_matched:
            continue   # already matched normally above - don't also reattach it to a stray candidate

        unassigned = np.nonzero(new_trackids == -1)[0]
        if unassigned.size == 0:
            break

        remaining_idxs = new_idxs[unassigned]                       #Shape (N_unassigned, 2)
        allowed_hops_cand, allowed_hops_dist = build_validjump_single(u, remaining_idxs, RMAX)   #Shape (N_unassigned, 2, N_neighbours)

        idx_arr = np.asarray(idx)
        stale_uni_idx = uni2oxy[idx_arr[idx_arr >= 0]]              #Shape (1,) or (2,), masked real atoms only

        inhop = (stale_uni_idx[None, :, None] == allowed_hops_cand) + (stale_uni_idx[None, ::-1, None] == allowed_hops_cand) #Shape (N_unassigned, N_atoms, N_neighbours)
        issource = inhop.any(axis=(1, 2))                           #Shape (N_unassigned,)

        if not issource.any():
            continue

        masked_dist = np.where(inhop, allowed_hops_dist, np.inf)    # Shape (N_unassigned, N_atoms, N_Neighbours)
        closest_dist = masked_dist.min(axis=(1, 2))                             # Compact down to (N_unassigned,)
        neighbour_rank = np.where(issource, masked_dist, np.inf)

        winner = np.argmin(neighbour_rank)
        new_trackids[unassigned[winner]] = tid

    #Then, assign new indices to latest
    unassigned = new_trackids == -1                                 # Remaining new_trackids elements
    spawn_qty = np.count_nonzero(unassigned)                        # Amount left to spawn in
    spawn_trackids = np.arange(next_id, next_id + spawn_qty)        # IDs for the tracks we need to spawn
    new_trackids[unassigned] = spawn_trackids                       # Assign new trackids
    next_id += spawn_qty                                            # Increment next available id

    #Refresh every track active this frame (matched normally, reattached above, or just spawned) to
    #a fresh (position, 0) lifetime entry - active tracks must never be treated as aging/stale.
    active_ids = set(new_trackids.tolist())
    active_lifetimes = {tid: (pos, 0) for tid, pos in zip(new_trackids, new_idxs.copy())}

    #Age every previously-known track NOT active this frame by one more missing frame, dropping it
    #once it exceeds the grace period.
    stale_lifetimes = {
        tid: (pos, stale_count + 1)
        for tid, (pos, stale_count) in old_lifetimes.items()
        if tid not in active_ids and stale_count + 1 < max_lifetime
    }

    all_lifetimes = {**stale_lifetimes, **active_lifetimes}

    return new_trackids, next_id, all_lifetimes


def run_multidefect_tracking(u_wrapped, tis, max_lifetime, RMAX = 5.0, isverbose = False):
    """
    Uses identify_frame_defects over a whole trajectory, which are then matched to previous
    existing defect tracks with match_defects.

    Returns a dict of numpy arrays (frame, track_id, type, atom1, atom2, position, type_names) -
    one row per (frame, active track), suitable for np.savez via save_tracks_npz.
    """
    pbar = tqdm if isverbose else __empty__

    # Build defect types and universe
    DFTYPES = ['OH', 'H3O', 'L', 'D']
    oxy_STATIC = u_wrapped.select_atoms("name O")
    box = u_wrapped.dimensions[:3]

    # Build mapping from oxygen indices to universe indices
    uni2oxy = np.full(len(u_wrapped.atoms)+1, -1, dtype = int)
    uni2oxy[oxy_STATIC.indices] = np.arange(len(oxy_STATIC))
    oxy2uni = np.concatenate([oxy_STATIC.indices, [-1]])

    # Initialize empty outputs
    next_id = {name : 0 for name in DFTYPES}
    old_idxs = {name : np.zeros((0,2), dtype = int) for name in DFTYPES}
    old_trackids = {name : np.zeros((0,), dtype = int) for name in DFTYPES}
    new_trackids = dict([(dfname, []) for dfname in DFTYPES])
    old_lifetimes = {name : {} for name in DFTYPES}

    #Hydrogen bond network info
    Nt = len(tis)
    Nh = len(u_wrapped.select_atoms("name H"))
    HBN_idx_save = np.full((Nt, Nh, 3), -1, dtype = int)
    HBN_ang_save = np.full((Nt, Nh), -1.0, dtype = float)

    dfsave = []

    for i, ts in enumerate(pbar(tis, desc = "Tracking defects")):
        u_wrapped.trajectory[ts]

        # Get ragged oxyNL
        oxy = u_wrapped.select_atoms("name O")
        oxyNL_ragged, oxyNL_weights = raw_oxyNeighbourList(oxy, box)

        # Get hydrogen bond neighbour network
        donor_O, accep_O, best_angles = get_hbond_neighbours(u_wrapped, oxyNL_ragged, oxyNL_weights)

        # Locate and classify all defects 
        classified = classify_defects(donor_O, accep_O, oxyNL_ragged, best_angles, oxy2uni)

        #Save HBN info
        hyd = u_wrapped.select_atoms("name H")
        oxy = u_wrapped.select_atoms("name O")

        HBN_idx_save[i, :, 0] = hyd.indices
        HBN_idx_save[i, :, 1] = oxy[donor_O].indices
        HBN_idx_save[i, :, 2] = oxy[accep_O].indices

        HBN_ang_save[i, :] = best_angles

        for type_code, name in enumerate(DFTYPES):

            #Extract useful data
            new_idxs = classified[name] if classified[name].size > 0 else np.full((0,2), -1, dtype = int)

            #Assign each dftype to a track index
            new_trackids[name], next_id[name], old_lifetimes[name] = match_defects(u_wrapped, new_idxs, old_idxs[name], old_trackids[name], old_lifetimes[name], next_id[name], max_lifetime, uni2oxy, RMAX)

            #Save output (not positions - depends on the way we identify D defects)
            for track_id, (atom1, atom2) in zip(new_trackids[name], new_idxs):
                dfsave.append((ts, track_id, type_code, atom1, atom2))

            #Update track indices
            old_idxs[name] = new_idxs
            old_trackids[name] = new_trackids[name]

    #Build dictionary outputs from ragged list
    frame, track_id, type_code, atom1, atom2 = zip(*dfsave) if dfsave else ([], [], [], [], [])

    defect_dict = {
        'frame' : np.array(frame, dtype = int),
        'track_id' : np.array(track_id, dtype = int),
        'type' : np.array(type_code, dtype = int),
        'atom1' : np.array(atom1, dtype = int),
        'atom2' : np.array(atom2, dtype = int),
        'type_names' : np.array(DFTYPES)
    }

    HBN_dict = {
        'tis' : tis,
        'hyd' : HBN_idx_save[:, :, 0],
        'donor' : HBN_idx_save[:, :, 1],
        'accep' : HBN_idx_save[:, :, 2],
        'HBN_angs' : HBN_ang_save
    }


    return defect_dict, HBN_dict

def identify_frame_defects_water(u, isverbose = False):
    """Identifies ionic defects in a water Universe"""

    oxy = u.select_atoms("name O")
    hyd = u.select_atoms("name H")

    #Useful quantities
    No = len(oxy)
    oind = np.arange(No)

    # Get hydrogen assignments
    hydoxyNL = get_hydNeighborList(oxy, hyd, u.dimensions, cutoff = 3.0)

    # Count assignments for each oxygen value
    counts = np.bincount(hydoxyNL, minlength = No)

    # Find defects
    classified = {
        "OH" : oind[counts == 1],
        "H3O" : oind[counts == 3]
    }
    located = {
        "OH" : [], 
        "H3O" : []
    }

    # Get positions and indices in dict form
    for name in ('OH', 'H3O'):
        for i in classified[name]:
            located[name].append((int(oxy[i].index), oxy[i].position.copy()))

    return located

def run_waterdefect_tracking(u_wrapped, tis, isverbose = False):
    """
    Uses identify_frame_defects_water over a whole trajectory.

    Returns a dict of numpy arrays (frame, type, atom_idx, position, type_names) -
    one row per identified atom in any frame, suitable for np.savez via save_tracks_npz.
    """
    pbar = tqdm if isverbose else __empty__

    # Relevant defect types
    DFTYPES = ['OH', 'H3O']

    # Universe properties
    oxy_STATIC = u_wrapped.select_atoms("name O")
    box = u_wrapped.dimensions[:3]

    # Index translator
    uni2oxy = np.full(len(u_wrapped.atoms)+1, -1, dtype = int)
    uni2oxy[oxy_STATIC.indices] = np.arange(len(oxy_STATIC))

    # Defect persistence matching parameters
    next_id = {name : 0 for name in DFTYPES}
    old_idxs = {name : np.zeros((0,2), dtype = int) for name in DFTYPES}
    old_trackids = {name : np.zeros((0,), dtype = int) for name in DFTYPES}
    new_trackids = dict([(dfname, []) for dfname in DFTYPES])
    old_lifetimes = {name : {} for name in DFTYPES}

    # Output list
    dfsave = []

    for i, ts in enumerate(pbar(tis, desc = "Tracking defects")):
        u_wrapped.trajectory[ts]

        #Note that anomalies here is in oxy indices
        located = identify_frame_defects_water(u_wrapped, isverbose = isverbose)

        for type_code, name in enumerate(DFTYPES):
            #Extract useful data
            if located[name]:
                idx_pairs, positions = zip(*located[name])
                new_idxs = np.array(idx_pairs, dtype = int)
                positions = np.array(positions)
            else:
                new_idxs = np.full((0,2), -1, dtype = int)
                positions = np.full((0,3), -1.0, dtype = float)

            #Save output (not positions - depends on the way we identify D defects)
            for atom, pos in zip(new_idxs, positions):
                dfsave.append((ts, type_code, atom))

            #Update track indices
            old_idxs[name] = new_idxs
            old_trackids[name] = new_trackids[name]

    #Build dictionary outputs from ragged list
    frame, type_code, atom = zip(*dfsave) if dfsave else ([], [], [])


    defect_dict = {
        'frame' : np.array(frame, dtype = int),
        'type' : np.array(type_code, dtype = int),
        'atom' : np.array(atom, dtype = int),
        'type_names' : np.array(DFTYPES)
    }

    return defect_dict

def vector_HBNN(u, donor_O, accep_O):
    oxy = u.select_atoms("name O")
    box = u.dimensions[:3]

    donor_pos = oxy[donor_O].positions
    accep_pos = oxy[accep_O].positions

    dists = accep_pos - donor_pos
    dists -= box[None, :] * np.rint(dists/box[None, :])

    vectors = np.zeros((donor_O.shape[0], 6))

    vectors[:, :3] = donor_pos
    vectors[:, 3:] = dists

    return vectors


def get_dist_pbc(pos1, pos2, box):
    """Gets the distance vector pointing from pos1 to pos2"""
    dist = pos2 - pos1
    dist -= box * np.rint(dist/box)
    return dist

def __empty__(x, **kwargs):
    return x

class NeighborNotFoundError(Exception):
    def __init__(self, *args):
        super().__init__(*args)