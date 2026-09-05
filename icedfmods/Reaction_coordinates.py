"""
This module contains all the ways I get reaction coordinates from my data
"""

import numpy as np

def get_ionic_delta(u, idxs, HBN_row):

    """
    Ionic reaction coordinates: 

    O* == H -- O'

    delta = |O*H| - |O'H|

    We choose our H to be the hydrogen bonded to O* such that delta is minimized
    """
    box = u.dimensions[:3]

    #When we dont find the defect, dont report delta
    dfidxs = idxs[idxs >= 0]
    if dfidxs.size < 1:
        return None, None    

    Hs_rows = np.nonzero(dfidxs[:, None] == HBN_row[1:, :])[1]   #indices of all rows where our defect appears
    Hs_idxs = HBN_row[0, Hs_rows].astype(int)

    #Get position of O* and all coordinated Hs
    dfpos = u.atoms[dfidxs[0]].position
    Hs_pos = u.atoms[Hs_idxs].positions

    #Get all distances from O* to its coordinated Hs
    OHdf_vecs = Hs_pos - dfpos[None, :]
    OHdf_vecs -= box * np.rint(OHdf_vecs/box)

    OHdf_dists = np.linalg.norm(OHdf_vecs, axis = -1)

    #Get the distances from the coordinated Hs to their other oxygen pair
    O_pairs_all = HBN_row[1:3, Hs_rows].astype(int)
    #Pick the non-defect oxygen per column
    is_donor_match = O_pairs_all[0] == dfidxs
    O_pairs = np.where(is_donor_match, O_pairs_all[1], O_pairs_all[0])

    OH_vecs = Hs_pos - u.atoms[O_pairs].positions
    OH_vecs -= box * np.rint(OH_vecs/box)

    OO_vecs = dfpos - u.atoms[O_pairs].positions
    OO_vecs -= box * np.rint(OO_vecs/box)

    OH_dists = np.linalg.norm(OH_vecs, axis = -1)
    OO_dists = np.linalg.norm(OO_vecs, axis = -1)

    #Use abs for consistency since for OH will be +ve but H3O will be -ve (test this)
    deltas = np.abs(OHdf_dists - OH_dists)
    mindelt = np.argmin(deltas)

    #Choose the O - H - O chain with the smallest delta (as in Markland/Marsalek)
    delta = (deltas[mindelt])
    oosum = (OO_dists[mindelt])

    #Output fraction of ionic defects not found for debugging
    #print(f'\n{invalid} / {Nt - start} ionics not found')

    return delta, oosum

def get_D_delta_atom(u, idxs, HBN_row):
    """
    This code measures the Bjerrum D reaction coordinate, defined by the the value
    cos(a1) - cos(a2), where a1 and a2 are the angles between the OO vector and both
    OH candidates.

    For example, in the sketch below:

    O          
          H*        H
         /         /
        O'   H^ - O"
            
    delta_D = cos(O'H*, O'O") - cos(O"H^, O'O")

    Then: 
        delta_D = 0: Symmetric D defect between O' and O"
        delta_D < 0: D defect is at O'
        delta_D > 0: D defect is at O" (not possible in this code)
    """

    dfidxs = idxs[idxs >= 0]

    #No defects, no delta (D defect needs 2)
    if dfidxs.size < 2:
        return None, None    
    
    #Find bond from dfidxs[0] -> dfidxs[1] and vice versa
    found_01 = (HBN_row[1, :] == dfidxs[0]) * (HBN_row[2, :] == dfidxs[1])
    found_10 = (HBN_row[1, :] == dfidxs[1]) * (HBN_row[2, :] == dfidxs[0])

    # Ensure there is only a single hydrogen pointing both ways
    if not(found_01.sum() * found_10.sum() == 1):
        return None, None

    #By construction, found_01 will be from O'H*
    alpha1 = HBN_row[3, found_01][0]
    alpha2 = HBN_row[3, found_10][0]

    delta = (alpha1 - alpha2)
    omega = (np.arccos(alpha1) + np.arccos(alpha2))

    return delta, omega

def get_D_delta_bond(u, start, idxs, HBNN, oxyNL):
    """
    TODO: Still uses legacy implementation doing loop inside fxn, maybe change this

    This code measures the Bjerrum D reaction coordinate, defined by the the value
    cos(a1) - cos(a2), where a1 and a2 are the angles between the OH vector and the
    two possible OO candidates

    For example, in the sketch below:

    O^          
          H*      H
         /       /
        O'  H - O"

    delta_D = cos(O'H*, O'O") - cos(O'H*, O'O^)

    Then: 
        delta_D = 0: H is suspended between OO bonds
        delta_D < 0: D defect is at O'O"
        delta_D > 0: D defect is at O'O^ (not possible in this code)
    """

    Nt = HBNN.shape[0]

    oxy_STATIC = u.select_atoms("name O")
    uni2oxy = np.zeros(len(u.atoms), dtype = int)
    uni2oxy[oxy_STATIC.indices] = np.arange(len(oxy_STATIC))
    oxy2uni = oxy_STATIC.indices

    box = u.dimensions[:3]

    delta = []
    omega = []

    for i, ts in enumerate(range(start, Nt)):
        u.trajectory[ts]

        dfidxs = idxs[i][idxs[i] >= 0]
        if dfidxs.size < 2: continue    #No defects, no delta (D defect needs 2)

        #Find possible H* (Hs) candidates (between the two defects)
        dfpair = np.vstack([dfidxs, dfidxs[::-1]], dtype = int)
        isbetween = ((HBNN[i, 1:3, :]).astype(int)[None, :, :] == dfpair[:, :, None]).all(axis = 1) #Shape (2, Nh) for acceptorind, Nh

        rows, cols  = np.nonzero(isbetween)

        try:
            acceptor_angles = HBNN[i, 3, cols] #shape (2,)
            H_idf = np.argmin(acceptor_angles)
            H_idx = HBNN[i, 0, cols[H_idf]].astype(int)
        except(ValueError):
            print(cols)
            print(np.count_nonzero(isbetween))
            print(dfpair)
            print(HBNN[i, 1:3, HBNN[i, 1, :] == dfidxs[0]])
            print(HBNN[i, 1:3, HBNN[i, 1, :] == dfidxs[1]])

            pairs_set = list(map(tuple, HBNN[i, 1:3, :].T))
            unique, counts = np.unique([pair[0] + 1000*pair[1] for pair in pairs_set], return_counts = True)
            pairs_set = set(pairs_set)
            reflected_mask = np.array([tuple(HBNN[i, 1:3:-1, k]) in pairs_set for k in range(HBNN.shape[-1])])
            print(70*"=")
            print(HBNN[i, 1:3, reflected_mask])
            print(f'{unique[counts > 1] % 1000}, {np.floor(unique[counts > 1] / 1000)}')

            continue

        #Find OO bonds for oxygen donating H_idx
        O_donor = HBNN[i, 1, cols[H_idf]].astype(int)

        O_NNs = oxy2uni[oxyNL[uni2oxy[O_donor], :]]

        #Neighbouring OO vectors from source to neighbour
        Od_pos = u.atoms[O_donor].position
        Os_vecs = u.atoms[O_NNs].positions - Od_pos[None, :]
        Os_vecs -=box[None, :]*np.rint(Os_vecs/box[None, :])

        #The relevant OH vector
        OH_vec = u.atoms[H_idx].position - Od_pos
        OH_vec -=box*np.rint(OH_vec/box)

        #Find magnitudes
        Os_dist = np.linalg.norm(Os_vecs, axis = -1)
        OH_dist = np.linalg.norm(OH_vec)

        #Find all angles
        prods = np.sum(OH_vec * Os_vecs, axis = -1)    #Shape (N_true,), matched to correct O
        angles = prods / (Os_dist * OH_dist)

        #For the defect OH*, alpha1 will be the second-best angle
        alpha1 = np.sort(angles)[-2]

        #Worst acceptor angle is alpha2
        alpha2 = np.min(acceptor_angles)

        delta.append(alpha1 - alpha2)
        omega.append(np.arccos(alpha1) + np.arccos(alpha2))

    return delta, omega

def get_L_delta(u, idxs, HBN_row):
    """
    This code measures the Bjerrum L reaction coordinate, defined by the the value
    cos(a1) - cos(a2), a1 is the angle between the incoming OH and the defect OO,
    while a2 is the angle between the incoming OH and its OO bond.

    For example, in the sketch below:

    O^     
          H*         H
          |         /
          O'      O"

    delta_L = cos(O'H*, O'O") - cos(O'H*, O'O^)

    Then: 
        delta_L = 0: H is suspended between OO bonds
        delta_L < 0: L defect is at O'O"
        delta_L > 0: L defect is at O'O^ (not possible in this code)
    """

    box = u.dimensions[:3]


    dfidxs = idxs[idxs >= 0]

    #No defects, no delta (L defect needs 2)
    if dfidxs.size < 2: 
        return None, None    
    #Find possible H* (Hs) candidates (where df is donor)
    found = dfidxs[:, None] == HBN_row[1, :].astype(int)[None, :]    #Shape (Ndf, Nh)
    rows, cols = np.nonzero(found)                           #Row is which Odf, col is which Hi
    Hs_idxs = HBN_row[0, cols].astype(int)
    dfO_idx = dfidxs[rows]                         #Attached O
    #Find other defect oxygen
    otherO_idx = np.where(dfO_idx == dfidxs[0], dfidxs[1], dfidxs[0])

    #Find OO vectors (pointing away from the O each H is attached to)
    dfO_pos = u.atoms[dfO_idx].positions        #Shape (N_true, 3)
    dfO_vec = u.atoms[otherO_idx].positions - dfO_pos
    dfO_vec -= box * np.rint(dfO_vec/box)

    #Find all OH vectors
    Hs_pos = u.atoms[Hs_idxs].positions         #Shape (N_true, 3)
    OH_vecs = Hs_pos- dfO_pos
    OH_vecs -= box[None, :]*np.rint(OH_vecs/box[None, :])

    #Find magnitudes
    dfO_dist = np.linalg.norm(dfO_vec, axis = -1)   #Should be the same everywhere tbh
    OH_dist = np.linalg.norm(OH_vecs, axis = -1)

    #Find all angles
    prods = np.sum(OH_vecs * dfO_vec, axis = -1)    #Shape (N_true,), matched to correct O
    angles = prods / (dfO_dist * OH_dist)

    #Most aligned OH bond to empty OO is the one we want
    maxind = np.argmax(angles)
    alpha1 = angles[maxind]

    #Then, we compare it against the bond score it has with its acceptor
    alpha2 = HBN_row[3, cols[maxind]]

    delta = (alpha1 - alpha2)
    omega = (np.arccos(alpha1) + np.arccos(alpha2))

    return delta, omega