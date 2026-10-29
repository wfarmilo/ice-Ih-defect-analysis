import numpy as np

def crosscorrFFT(a, b):
    N = len(a)
    A = np.fft.fft(a, n = 2*N)
    B = np.fft.fft(b, n = 2*N)
    PSD = A.conjugate() * B
    res = np.fft.ifft(PSD)
    ccf = (res[:N]).real
    return ccf

def msd_fft_1d_masked(r, mask):
    rm = r * mask
    qm = mask * r**2
    S1 = crosscorrFFT(mask, qm) + crosscorrFFT(qm, mask)
    S2 = crosscorrFFT(rm, rm)   # Autocorrelation fxn
    count = np.rint(crosscorrFFT(mask, mask))
    msd = np.full(r.shape, np.nan, dtype = float)
    valid = (count != 0)
    msd[valid] = (S1[valid] - 2*S2[valid])/count[valid]
    return msd, count

def masked_fft_msd(defectpos : np.ndarray[float], mask : np.ndarray[int], unwrap : bool = False, box : np.ndarray[float] = None):

    if unwrap:
        assert box is not None, f'Box {box} not specified for unwrap = True.'
        defectpos_final = np.zeros(defectpos.shape)
        seen = mask == 1
        defectpos_final[seen] = unwrap_defect(defectpos[seen], box)
    else:
        defectpos_final = defectpos.copy()

    msd = np.zeros([defectpos.shape[0], 3])
    for i in range(3):
        
        msd_temp, count = msd_fft_1d_masked(defectpos_final[:, i], mask)

        msd[:, i] += msd_temp

    return msd, count

def unwrap_defect(defect, box):

    df = defect.copy()

    for i in range(df.shape[0]-1):
        old = df[i, :]
        new = df[i+1, :]

        dist = new - old
        dist -= box * np.rint(dist/box)

        df[i+1, :] = old + dist

    return df

def blank_frame_positions(u, name, frames, atom1, atom2, MIDPOINT_DEFECT = ["L"]):
    """
    Method to fill in frames where defects do not exist with zeros, and extract their 
    positions when found. Will extend the defect to fill between frames[0] and frames[-1], 
    NOT the entire universe trajectory. Will use atom1 positions for all defects, unless 
    their name is present in MIDPOINT_DEFECT (default ["L"]) in which case the pbc-aware 
    midpoint between atom1 and atom2 will be considered.
    """

    # If given an empty array, return emptiness
    if frames.size < 1:
        return np.zeros(0, dtype = int), np.zeros((0,3)), np.zeros(0, dtype = int)

    t0 = frames[0]
    tf = frames[-1]
    Nt = tf - t0
    frame_range = np.arange(t0, tf)
    box = u.dimensions[:3]
    pos = np.zeros((Nt, 3), dtype = float)
    valid_frames = np.ones(Nt, dtype = int)


    frame_counter = 0
    for ti, ts in enumerate(np.arange(t0, tf)):
        u.trajectory[ts]

        if not(ts == frames[frame_counter]):
            pos[ti, :] = np.zeros(3)
            valid_frames[ti] = 0
            continue

        idxs = np.array([atom1[frame_counter], atom2[frame_counter]])
        idxs = idxs[idxs >= 0]

        frame_counter += 1

        if name in MIDPOINT_DEFECT:
            a1 = u.atoms[idxs[0]].position
            a2 = u.atoms[idxs[1]].position
            dist = a2 - a1
            dist -= box * np.rint(dist/box)
            pos[ti, :] = a1 + dist/2
        else:
            pos[ti, :] = u.atoms[idxs[0]].position
    return frame_range, pos, valid_frames