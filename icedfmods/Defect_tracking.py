import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances as mddist
from tqdm import tqdm
import freud as fd
from scipy.optimize import milp, LinearConstraint, Bounds, linear_sum_assignment
from scipy.sparse import coo_matrix

#This builds on the framework of Neighbor_mod.py, but should be more robust (hopefully)
#
#Unlike Multidefect_tracking_mod.py (which this started as a copy of), the classification below
#can identify an unbounded number of simultaneous OH-/H3O+/L/D defects per frame, and DefectTracker
#gives each one a stable identity across frames instead of relying on raw index order (which flips
#whenever two same-type defects cross in sorted-index order).

def get_oxyNeighborList(u):
    #Define useful params
    oxy = u.select_atoms('name O')
    No = len(oxy)
    box = fd.box.Box(*u.dimensions[:3]) #Assumes orthogonal box
    points = oxy.positions

    voro = fd.locality.Voronoi()

    voro.compute((box, points))

    nlist = voro.nlist

    i = nlist.query_point_indices
    j = nlist.point_indices
    d = nlist.distances
    w = nlist.weights

    pairs = np.zeros((No, No), dtype = float)
    pairs[i, j] = w/d   #Enhanced weighting by inverse distance

    #Symmetrize weights (each (i,j) has one maximized w)
    pairs = np.maximum(pairs, pairs.T)

    #Get indices for upper triangular part of pairs
    iu, ju = np.triu_indices(No, k=1)

    #Keep only bonded pairs
    valid = pairs[iu, ju] > 0
    iu_valid = iu[valid]
    ju_valid = ju[valid]
    wu_valid = pairs[iu_valid, ju_valid]
    
    N_edge = len(wu_valid)

    #The goal here is to maximize the matrix product weights X keep_bond
    minimize = -wu_valid

    #Establish contact matrix: contact[Oi, Ek] = 1 if oxygen Oi is on the edge Ek
    rows = np.concatenate((iu_valid, ju_valid))                     #oxygen index from iu and ju (will end up symmetric)
    cols = np.concatenate((np.arange(N_edge), np.arange(N_edge)))   #Edge index just increases
    connected = np.ones(2*N_edge, dtype = int)                      #Value to fill at that point
    contact = coo_matrix((connected, (rows, cols)), shape = (No, N_edge))

    mustbefour = LinearConstraint(contact, 4, 4)    #Must be between 4 and 4 (so = 4) contacts

    #Set up constraints on my solution x, which tells me to keep (x[k] = 1) or discard (x[k] = 0) bond k
    bounds = Bounds(0,1)                        # Enforce above, must be between 0 and 1
    isinteger = np.ones(N_edge, dtype = int)    # Must also be an integer

    result = milp(
        c = minimize,               #The coefficients to be minimized
        integrality = isinteger,    #Enforce the solution to be an integer
        bounds = bounds,            #Enforce the solution to be 0 or 1
        constraints = mustbefour    #Enforce that each Oi have 4 neighbours
    )

    assert result.success, result.message

    keep_bond = result.x > 0.5

    NNlist = [[] for _ in range(No)]

    for Oi, Oj in zip(iu_valid[keep_bond], ju_valid[keep_bond]):
        NNlist[Oi].append(Oj)
        NNlist[Oj].append(Oi)

    NNind = np.array(NNlist, dtype = int)
    
    return NNind

def _find_validjumps(oxy, box, oxyNL, max_NN_hop = 2):
    No = len(oxy)

    validjump = [set([i]) for i in range(No)]
    jumplists = []

    for i in range(No):
        for n in range(max_NN_hop):
            neighbours = oxyNL[list(validjump[i])].flatten()
            validjump[i] |= set(neighbours.tolist())

        #Cast back to arr for easy indexing
        jumplist = np.array(list(validjump[i]), dtype = int)

        #Sort by distance from source
        dists = np.linalg.norm(get_dist_pbc(oxy.positions[i], oxy.positions[jumplist], box), axis = -1)
        jumplists.append(jumplist[np.argsort(dists)])

    #Per-oxygen reachable-neighbor counts aren't guaranteed equal (depends on local ring
    #structure, especially near a defect), so pad to a common width with -1 instead of assuming
    #a rectangular shape.
    max_len = max(len(j) for j in jumplists)
    padded = np.full((No, max_len), -1, dtype = int)
    for i, j in enumerate(jumplists):
        padded[i, :len(j)] = j

    #Stored as (No+1, N_neighbors), last row is for -1 flags
    return np.vstack([padded, -1 * np.ones((1, max_len), dtype = int)])


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

def get_align_HBNN(u, oxyNL):
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
    NN_O_idx = oxyNL[hydoxyNL, :]   #shape (Nh, 4)
    NN_O_pos = oxy.positions[NN_O_idx] #shape will be (Nh, 4, 3) for [Hk, Oj, xyz]

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
    prod = np.einsum('ij,ikj->ik', OH_vecs, OO_vecs)    #Shape (Nh, 4)

    sorted_idx = np.argsort(prod, axis = 1)

    best_O = sorted_idx[:, -1]

    best_angles = prod[np.arange(Nh), best_O]
    delta = best_angles - prod[np.arange(Nh), sorted_idx[:, -2]]
    #avg = np.mean(best_angles)
    #std = np.std(best_angles)

    #valid = best_angles >= (avg - 2*std)       #Filter only valid bonds


    accep_O = NN_O_idx[np.arange(Nh), best_O]   #Shape (Nh,)
    donor_O = hydoxyNL                          #Shape (Nh,)


    return donor_O, accep_O, best_angles, delta

def recalibrate_HBNN(u, L_idx, oxyNL, isverbose):

    #Find list of unique indices
    err_idx = np.unique(np.concatenate((L_idx, oxyNL[L_idx, :].flatten())))

    #Useful params
    oxy = u.select_atoms("name O")
    hyd = u.select_atoms("name H")
    box = u.dimensions[:3]
    No = len(oxy)
    Nh = len(hyd)

    err_map = np.zeros(No, dtype = int)
    err_map[err_idx] = np.arange(len(err_idx))  #Maps oxygen index to err index

    #Find oxygen neighbours by distance
    dists = mddist.distance_array(oxy[err_idx], oxy, box = u.dimensions)
    mindist = np.argsort(dists, axis = 1)

    errNL = mindist[:, 1:7]

    #Get hydrogens associated with elements in err
    hydoxyNL = get_hydNeighborList(oxy, hyd, u.dimensions, cutoff = 3.0)

    err_hyds = np.arange(Nh)[((hydoxyNL[:, None] == err_idx[None, :])).any(axis=1)]    #List of all Hs attached to an O in err
    err_hydoxy = hydoxyNL[err_hyds]                                                    #List of which O is attached to err_hyds[k]
    Nh_err = len(err_hyds)                                                             #How many Hs we found (usually 2x len(err_idx), but may be more/less)

    #Find OH vectors
    OH_pos = oxy[err_hydoxy].positions
    OH_vecs = hyd[err_hyds].positions - OH_pos
    OH_vecs -= box[None, :] * np.rint(OH_vecs/box[None, :])

    #Find all O neighbour positions
    NN_O_idx = errNL[err_map[err_hydoxy], :]
    NN_O_pos = oxy.positions[NN_O_idx]  #shape will be (Nh_err, 6, 3)

    if isverbose: print('\n'.join([f'{err} : {errN}' for err, errN in zip(oxy[err_idx].indices, oxy.indices[errNL])]))

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
    #   k: oxygen neighbour index (0..5)
    # Then, what this sum is doing is for each for each element ij in OH_vecs, it 
    # adds the element ikj in OO_vecs and stores it in element ik of prod. Thus,
    # we sum over all of the xyz coordinates j of the product (denoted by ',' here)
    prod = np.einsum('ij,ikj->ik', OH_vecs, OO_vecs)

    sorted_idx = np.argsort(prod, axis = 1)

    best_O = sorted_idx[:, -1]

    accep_O = NN_O_idx[np.arange(Nh_err), best_O]   #Shape (Nh_err,)
    donor_O = err_hydoxy                            #Shape (Nh_err,)

    return err_hyds, donor_O, accep_O



def classify_defects(donor_O, accep_O, oxyNL, bestangles):
    """
    Identifies every OH-/H3O+/L/D defect candidate present this frame - no cap on how many of
    each type may coexist, unlike get_df_HBNN's single-instance-oriented logic.

    Returns (classified, anomalies). classified['OH']/['H3O'] are 1-D oxygen-index arrays;
    classified['L']/['D'] are (n,2) arrays of oxygen-index pairs flanking each broken (L) or
    doubly-occupied (D) edge. anomalies flags candidate sites where identification was ambiguous
    (see _find_edge_defects), so a caller can retry (e.g. via recalibrate_HBNN) instead of
    silently mispairing or misclassifying.
    """
    No = oxyNL.shape[0]
    oind = np.arange(No)

    #Robust count of which oxygen is occupied
    donor_cts = np.bincount(donor_O, minlength = No)
    accep_cts = np.bincount(accep_O, minlength = No)
    total_cts = donor_cts + accep_cts

    #Use coordination for ionic
    OH_idx = oind[donor_cts == 1]
    H3O_idx = oind[donor_cts == 3]

    #Builds oxygen-ordered dict of edges, keyed by No * Oi + Oj 
    edge_counts = _build_edge_counts(donor_O, accep_O, No)

    #L and D are topological mirrors of each other on the same edge-count structure: L is a missing
    #(0-hydrogen) edge, D is a doubly-occupied (2-hydrogen) edge - see _build_edge_counts.
    L_pairs, L_anomalies = _find_edge_defects(oind[total_cts == 3], oxyNL, No, edge_counts, target_count = 0)
    D_pairs, D_anomalies = _find_edge_defects(oind[total_cts >= 5], oxyNL, No, edge_counts, target_count = 2)

    #Sort so that if we want to use the localized index we use the first
    D_pairs_sorted = _reorder_D_pairs(donor_O, accep_O, D_pairs, bestangles)

    classified = {
        'OH': OH_idx,
        'H3O': H3O_idx,
        'L': L_pairs,
        'D': D_pairs_sorted,
    }
    anomalies = {'L': L_anomalies, 'D': D_anomalies}

    return classified, anomalies


def _build_edge_counts(donor_O, accep_O, No):
    """
    Counts how many hydrogens are assigned along each O-O edge, keyed by a direction-independent
    canonical key (min(i,j)*No + max(i,j)) so a bond i->j and a bond j->i along the same edge count
    as the same edge rather than two different ones. Every hydrogen contributes exactly 1 to its
    own edge's count (canon has one entry per hydrogen), so: a normal (singly-occupied) edge has
    count 1, a missing edge (L-defect: neither flanking oxygen's H points along it) has count 0,
    and a doubly-occupied edge (the traditional Bjerrum D-defect: both flanking oxygens' hydrogens
    point at each other along the same O-O axis) has count 2.
    """
    a = donor_O.astype(np.int64)
    b = accep_O.astype(np.int64)
    edge_key = np.minimum(a, b) * No + np.maximum(a, b) #A bond from a->b == b->a (and has count 2), shape Nh
    edge_ids, counts = np.unique(edge_key, return_counts = True)
    return dict(zip(edge_ids.tolist(), counts.tolist()))


def _find_edge_defects(candidates, oxyNL, No, edge_counts, target_count):
    """
    For each candidate oxygen, finds which of its 4 oxyNL neighbours sits on an edge with exactly
    `target_count` hydrogens assigned (0 = missing/L-defect, 2 = doubly-occupied/D-defect)
    A candidate with zero or 2+ matching neighbours is ambiguous and reported
    instead of guessed at.
    """
    pairs = []
    seen_edges = set()
    anomalies = []

    for Oi in candidates:
        Oi = int(Oi)
        matches = [int(Oj) for Oj in oxyNL[Oi]  #Check neighbours of Oi
                   if edge_counts.get(min(Oi, int(Oj)) * No + max(Oi, int(Oj)), 0) == target_count] #Only flag as match if the edge count is our target (if not found, is 0)

        if len(matches) == 1:
            #Label the edge sorted by index (permutation-invariant)
            edge = (min(Oi, matches[0]), max(Oi, matches[0]))
            #Both oxygens flanking a defect edge satisfy the same coordination filter, so this
            #edge is typically found once from each end - only keep it once.
            if edge not in seen_edges:
                seen_edges.add(edge)
                pairs.append(edge)
        else:
            #If more than 1 edge defect on a site, report it
            anomalies.append((Oi, matches))

    #Return pairs (if they exist) and anomalies
    pairs_arr = np.array(pairs, dtype = int) if pairs else np.full((0,2), -1, dtype = int)
    return pairs_arr, anomalies

def _reorder_D_pairs(donor_O, accep_O, D_pairs, bestangles):
    """
    Reorders D defect pairs so that the first entry is the one which would be identified as a 
    single D defect site if we use the oxygen site definition instead of the bond definition.
    """

    D_pairs_sorted = []
    for pair in D_pairs:
        #D_pairs is guaranteed sorted by lowest index -> highest index by construction
        first = (donor_O == pair[0]) * (accep_O == pair[1])
        second = (donor_O == pair[1]) * (accep_O == pair[0])

        #If we have a double donating/accepting D defect, kill it (artifact from oxyNL issues)
        if not(first.any()) or not(second.any()):
            return np.full((0,2), -1, dtype = int)

        angs = [bestangles[first][0], bestangles[second][0]]

        #Smallest angs value is dangling H-bond
        sortinds = np.argsort(angs)

        D_pairs_sorted.append(pair[sortinds])

    return D_pairs_sorted

def locate_defect_positions(u, classified):
    """
    Converts classify_defects' positional oxygen/hydrogen indices into real universe atom indices
    and PBC-aware positions, one entry per candidate - not one PBC-blended average per type like
    the old get_dfpos, which collapses multiple simultaneous same-type defects into a single
    position that corresponds to neither of them.
    """
    oxy = u.select_atoms("name O")
    hyd = u.select_atoms("name H")
    box = u.dimensions[:3]

    located = {'OH': [], 'H3O': [], 'L': [], 'D': []}

    for name in ('OH', 'H3O'):
        for i in classified[name]:
            located[name].append(((int(oxy[i].index), -1), oxy[i].position.copy()))

    #L and D are both flanking-oxygen-pair edges (missing/doubly-occupied respectively) - same
    #midpoint-of-edge position convention for both.
    for name in ('L', 'D'):
        for i, j in classified[name]:
            vec = get_dist_pbc(oxy[i].position, oxy[j].position, box)
            midpoint = oxy[i].position + vec / 2.0
            located[name].append(((int(oxy[i].index), int(oxy[j].index)), midpoint))

    return located


def identify_frame_defects(u, oxyNL, donor_O, accep_O, best_angles, recalibrate = True, isverbose = False):
    """
    Runs one frame's full identification pipeline: build the HBNN, classify all candidates, check
    for and patch any stale oxyNL rows near this frame's candidates (see _find_stale_oxyNL_rows;
    oxyNL is patched in place, so a fix persists for every later frame the caller passes the same
    array to - not just this one), then (once) retry via recalibrate_HBNN for any L site whose
    missing edge is still ambiguous after that - generalizing multidefect_tracking.py's old
    `len(L_idx) > 2`-triggered recalibration (which always re-solved the whole current L_idx set)
    to fire only on the specific sites that are actually ambiguous, correctly scoped for any number
    of concurrent defects.
    """

    #Get defects from donor counts (ionic) and edge tracking (bjerrum)
    classified, anomalies = classify_defects(donor_O, accep_O, oxyNL, best_angles)

    #If broken, try a one-shot fix
    if recalibrate and (anomalies['L'] or anomalies['D']):
        err_idx = np.array(sorted({i for i, _ in anomalies['L']}), dtype = int)
        fix_idx, fix_donor, fix_accep = recalibrate_HBNN(u, err_idx, oxyNL, isverbose)

        donor_O = donor_O.copy()
        accep_O = accep_O.copy()
        donor_O[fix_idx] = fix_donor
        accep_O[fix_idx] = fix_accep

        classified, anomalies = classify_defects(donor_O, accep_O, oxyNL, best_angles)

    located = locate_defect_positions(u, classified)
    return located, anomalies

def match_defects(validjump, new_idxs, old_idxs, old_trackids, old_lifetimes, next_id, max_lifetime, uni2oxy):

    N_new = len(new_idxs)
    N_old = len(old_idxs)
    N_neighbours = validjump.shape[-1]

    #Output: new ids for each oxygen
    new_trackids = np.full(N_new, -1, dtype = int)

    #Array of valid candidates for each old atom (in oxygen indices)
    allowed_hops = validjump[uni2oxy[old_idxs], :]   #Shape N_old, N_atoms, N_neighbours

    #Boolean of possible sources for each atom
    possible_sources = np.zeros((N_new, N_old), dtype = bool)
    distance_sources = np.full((N_new, N_old), N_neighbours, dtype = int)

    #Indexes (in new/old_idxs)
    Old, New = np.meshgrid(np.arange(N_old), np.arange(N_new))

    #For each new candidate, check possible sources
    for ni in range(N_new):
        mask_idxs = new_idxs[ni, :] >= 0
        new_uni_idx = uni2oxy[new_idxs[ni, mask_idxs]]  #Shape N_atoms (2 or 1)
        inhop = (new_uni_idx[None, :, None] == allowed_hops) + (new_uni_idx[None, ::-1, None] == allowed_hops) #shape (N_old, N_atoms, N_neghbours)
        issource = inhop.any(axis=(1,2))                            #shape N_old
        closest_neighbour = np.argmax(inhop.any(axis=1), axis = 1)  #shape N_old, indexing N_neighbours (.any() collapses N_atoms, argmax finds first True among neighbours)
        neighbour_rank = np.where(issource, closest_neighbour, N_neighbours)    #Where issource, output closest_neighbour value otherwise use large flag N_neighbours

        possible_sources[ni] = issource
        distance_sources[ni] = neighbour_rank

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
        closest_by_output = np.where(oneside, distance_sources, N_neighbours)   #masked array of shape (N_new, N_old) with distance ranks filled in where one-sided pairs exist
        winner_by_output = np.argmin(closest_by_output, axis = 0)               #Find closest paired output for each source (shape N_old)
        source_winner_ispaired = oneside.any(axis = 0)                          #Per source, is any output trying to pair with it

        #Construct matrix of winning sources for each conflicting output
        source_winners = np.zeros((N_new, N_old), dtype = bool)
        source_winners[winner_by_output[source_winner_ispaired], np.arange(N_old)[source_winner_ispaired]] = True   #Now each source has at most one output paired with it

        #Resolve shared source overlap
        closest_by_source = np.where(source_winners, distance_sources, N_neighbours) #masked array of shape (N_new, N_old) with distance ranks filled in where we have source winners
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

    #Fallback: leftover ambiguity the count-based peel couldn't break (e.g. a symmetric tie) gets
    #resolved by nearest hop-distance, same escalation DefectTracker._step_type uses.
    remaining_new = np.nonzero(possible_sources.any(axis = 1))[0]
    remaining_old = np.nonzero(possible_sources.any(axis = 0))[0]
    if remaining_new.size and remaining_old.size:
        sub_possible = possible_sources[np.ix_(remaining_new, remaining_old)]
        sub_cost = np.where(sub_possible, distance_sources[np.ix_(remaining_new, remaining_old)], N_neighbours + 1)
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
        allowed_hops_cand = validjump[uni2oxy[remaining_idxs], :]   #Shape (N_unassigned, 2, N_neighbours)

        idx_arr = np.asarray(idx)
        stale_uni_idx = uni2oxy[idx_arr[idx_arr >= 0]]              #Shape (1,) or (2,), masked real atoms only

        inhop = (stale_uni_idx[None, :, None] == allowed_hops_cand) + (stale_uni_idx[None, ::-1, None] == allowed_hops_cand) #Shape (N_unassigned, N_atoms, N_neighbours)
        issource = inhop.any(axis=(1, 2))                           #Shape (N_unassigned,)

        if not issource.any():
            continue

        closest_neighbour = np.argmax(inhop.any(axis=1), axis=1)    #Shape (N_unassigned,)
        neighbour_rank = np.where(issource, closest_neighbour, N_neighbours)

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


def run_multidefect_tracking(u_wrapped, tis, max_lifetime, isverbose = False, recalibrate = True):
    """
    Drives DefectTracker over a whole trajectory, replacing multidefect_tracking.py's fixed-shape
    defectidx/defectpos accumulation loop. `u_wrapped` must already have its trajectory wrapped
    (compound='atoms'), the same way multidefect_tracking.py sets it up before its main loop.

    Returns a dict of numpy arrays (frame, track_id, type, atom1, atom2, position, type_names) -
    one row per (frame, active track), suitable for np.savez via save_tracks_npz.
    """
    pbar = tqdm if isverbose else __empty__

    DFTYPES = ['OH', 'H3O', 'L', 'D']
    oxy_STATIC = u_wrapped.select_atoms("name O")
    box = u_wrapped.dimensions[:3]

    uni2oxy = np.full(len(u_wrapped.atoms)+1, -1, dtype = int)
    uni2oxy[oxy_STATIC.indices] = np.arange(len(oxy_STATIC))

    oxyNL = get_oxyNeighborList(u_wrapped)
    validjump = _find_validjumps(u_wrapped.select_atoms("name O"), box, oxyNL, max_NN_hop=2)

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

        #Get hydrogen bond neighbour network
        donor_O, accep_O, best_angles, _ = get_align_HBNN(u_wrapped, oxyNL)

        #Note that anomalies here is in oxy indices
        located, anomalies = identify_frame_defects(u_wrapped, oxyNL, donor_O, accep_O, best_angles, recalibrate = recalibrate, isverbose = isverbose)

        #Save HBN info
        hyd = u_wrapped.select_atoms("name H")
        oxy = u_wrapped.select_atoms("name O")

        HBN_idx_save[i, :, 0] = hyd.indices
        HBN_idx_save[i, :, 1] = oxy[donor_O].indices
        HBN_idx_save[i, :, 2] = oxy[accep_O].indices

        HBN_ang_save[i, :] = best_angles

        for type_code, name in enumerate(DFTYPES):
            #Extract useful data
            if located[name]:
                idx_pairs, positions = zip(*located[name])
                new_idxs = np.array(idx_pairs, dtype = int)
                positions = np.array(positions)
            else:
                new_idxs = np.full((0,2), -1, dtype = int)
                positions = np.full((0,3), -1.0, dtype = float)

            #Assign each dftype to a track index
            new_trackids[name], next_id[name], old_lifetimes[name] = match_defects(validjump, new_idxs, old_idxs[name], old_trackids[name], old_lifetimes[name], next_id[name], max_lifetime, uni2oxy)

            #Save output (not positions - depends on the way we identify D defects)
            for track_id, (atom1, atom2), pos in zip(new_trackids[name], new_idxs, positions):
                dfsave.append((ts, track_id, type_code, atom1, atom2))

            #Update track indices
            old_idxs[name] = new_idxs
            old_trackids[name] = new_trackids[name]
        

        if isverbose and anomalies['D']:
            print(f"\nFrame {ts}: {len(anomalies['D'])} ambiguous D-candidate site(s) skipped: {anomalies['D']}")
        if isverbose and anomalies['L']:
            print(f"\nFrame {ts}: {len(anomalies['L'])} ambiguous L-candidate site(s) after recalibration: {anomalies['L']}")

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