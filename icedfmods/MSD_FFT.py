"""
This module contains functions for computing the mean-squared displacement
by means of a fast fourier transform, to go from O(N^2) complexity in the
standard approach to O(NlogN). Most of these functions were taking from Prof.
Kara Fong's repository, https://github.com/kdfong/transport-coefficients-MSD
"""


import numpy as np

#Fast fourier transform diffusion stuff
def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)        #Fast fourier transform, with output length 2N
    PSD = F * F.conjugate()         #Power spectral density (mean square value of fourier components)
    res = np.fft.ifft(PSD)          #Inverse fourier transform of power spectral density -> autocorrelation
    res= (res[:N]).real             #Cleans up the imaginary parts from approximation (probably)
    n=N*np.ones(N)-np.arange(0,N)   
    acf = res/n                     #Autocorrelation function?
    return acf

def msd_fft_1d(r):
    N=len(r)
    D=np.square(r)
    D=np.append(D,0)    #Array of squares now has 0 at end (can evaluate D[N])
    S2=autocorrFFT(r)   #Position autocorrelation
    Q=2*D.sum()         #2*sum of squares
    S1=np.zeros(N)      
    for m in range(N):      
        Q=Q-D[m-1]-D[N-m]   #From the first sum of squares, remove the first value (sum from 0 to N-m, offset by m -> sum_0^{N-m}(r^2_{m+1})). From the 2nd, remove the first value (sum from 0 to N-m)
        S1[m]=Q/(N-m)       #Divide by N-m to get delta
    return S1-2*S2          #Subtract 2*autocorrelation for a given lag time m (middle term in |<r(t)> - <r(0)>|^2)

def get_fft_msd(defect, unwrapped = False, box = None):
    """ 
    Xavi's method for finding the MSD using a Fast Fourier Transform from Kara's library. Currently only works for 1 defect/timestep.
    
    Parameters: 
    - defect: numpy array of defect positions with columns (x, y, z). Shape (Nt, 3)
    - unwrapped: boolean which indicates whether the defect positions have been unwrapped from PBCs
    - box: pbc box for orthorhombic cell, of the form (Lx, Ly, Lz)

    Returns:
    - msd: per-axis mean-squared displacwement

    """

    #Unwrap positions if not already done
    if not(unwrapped):
        defect_unwrapped = unwrap_defect(defect, box)
    else:
        defect_unwrapped = defect.copy()

    msd = np.zeros([defect.shape[0], 3]) 

    for i in range(3):  # iterate over x, y, z for all times
        msd_temp = msd_fft_1d(np.array(defect_unwrapped[:, i]))

        # Ensure temporary msd is appropriately sized
        msd[:, i] = msd_temp

    return msd

def unwrap_defect(defect, box):

    df = defect.copy()

    for i in range(df.shape[0]-1):
        old = df[i, :]
        new = df[i+1, :]

        dist = new - old
        dist -= box * np.rint(dist/box)

        df[i+1, :] = old + dist

    return df

def get_frame_positions(u, name, frames, atom1, atom2):
    """
    Method to fill in frames where defects, and extract their positions when found
    """

    Nt = len(u.trajectory)
    box = u.dimensions[:3]
    pos = np.zeros((Nt, 3), dtype = float)

    # Works since frames is increasing by construction
    frame_lens = np.concatenate(([frames[0]], frames[1:] - frames[:-1], [Nt - frames[-1]]))

    indices = np.vstack([atom1, atom2]).T   # Shape (Nf, 2)
    indices_Nt = np.repeat(indices, frame_lens, axis = 0)   # Shape (Nt, 2)

    for ti in range(Nt):
        u.trajectory[ti]
        idxs = indices_Nt[ti, :]
        idxs = idxs[idxs > 0]

        if name != "L":
            pos[ti, :] = u.atoms[idxs[0]].position
        else:
            a1 = u.atoms[idxs[0]].position
            a2 = u.atoms[idxs[1]].position
            dist = a2 - a1
            dist -= box * np.rint(dist/box)
            pos[ti, :] = a1 + dist/2
    return pos