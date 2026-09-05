import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

"""
Helper module for plot_results.py containing the different types of plot
as perscribed by the plot_style JSON input under the "type" field

"""

# Order of plot styles
DEF_STYLE = {"c" : "k", "ls" : "-", "dash_capstyle" : "round"}
LS_ORDER = ["-", ":", "--", "-."]
def CLR_ORDER(num_keys, cmap): return mpl.colormaps[cmap](np.linspace(0, 0.9, num_keys))

def get_keytoorder(isostyle, isolated, cmap = "viridis"):
    try:
        color_key = list(isostyle.keys())[list(isostyle.values()).index("c")]
        num_clrs = len(isolated[color_key])
    except(ValueError):
        # Color not needed, this doesnt matter
        num_clrs = 2   

    return {"ls" : LS_ORDER, "c" : CLR_ORDER(num_clrs, cmap)}

def plot_free1d(ax, isostyle, isolated, nametomap, delta, d_sum, inputs, nbins, prefac, cmap = "viridis"):

    # Get histogram bins
    bins = np.linspace(-1, 1, nbins)
    bins_centered = (bins[1:] + bins[:-1])/2

    # Turn pooled valid deltas and sums into histogram
    data = delta / d_sum
    mirrored = np.concatenate([-data, data])
    hist, _ = np.histogram(mirrored, bins = bins, )

    # Get free energy from histogram (normalized by total summed runs)
    free = - prefac * np.log(hist/hist.sum())
    finite = np.isfinite(free)

    # Shift down
    free -= np.min(free[finite])

    # Tells you the value in inputs (should be repeated the whole time since these are isolated vars) corresponding to the style of that thing
    style = [(inputs[0, nametomap[key]], isostyle[key]) for key in sorted(isolated)]

    # Label from key to ordering
    keytoorder = get_keytoorder(isostyle, isolated, cmap)

    styleparams = dict([(s[1], '') for s in style])
    for i, key in enumerate(sorted(isolated)):
        styleparams[style[i][1]] = keytoorder[style[i][1]][list(map(str, isolated[key])).index(style[i][0])]

    # Plot
    updparams = DEF_STYLE.copy() | styleparams
    line, = ax.plot(bins_centered, free, **updparams)

    # Label
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\mathcal{F}$ [$k_BT$]")

    ax.set_xlim([-0.25, 0.25])

    return line

def plot_free2d(ax, isostyle, isolated, nametomap, delta, d_sum, inputs, nbins, prefac):

    # Get boundaries
    maxabs_delta = np.max(np.abs(delta))
    max_sums = np.max(d_sum)
    min_sums = np.min(d_sum)

    # Get bins
    bins_delta = np.linspace(-maxabs_delta, maxabs_delta, nbins)
    bins_d_sum = np.linspace(min_sums, max_sums, nbins)

    # Get mirrored histogram
    mirror_delta = np.concatenate([-delta, delta])
    mirror_d_sum = np.concatenate([d_sum, d_sum])

    hist, _, _ = np.histogram2d(mirror_delta, mirror_d_sum, bins = [bins_delta, bins_d_sum])
    with np.errstate(divide='ignore', invalid = 'ignore'):  #Ignore warnings
        free = -prefac*np.log(hist/hist.sum())

    # Shift down, and filter infinities to blank NaN values
    finite = np.isfinite(free)
    free -= np.min(free[finite])
    free[np.logical_not(finite)] = np.nan

    # Plot
    cmap = plt.get_cmap('viridis').copy()
    norm2d = plt.Normalize(vmin = 0, vmax = np.max(free)) #Sets colormap scale

    image, = ax.imshow(
        free.T,
        cmap = cmap,
        norm = norm2d,
        aspect = 'auto', 
        origin = 'lower', 
        extent = (bins_delta[0], bins_delta[-1], bins_d_sum[0], bins_d_sum[-1])
        )

    return image

def plot_barrier(ax, isostyle, isolated, nametomap, delta, d_sum, inputs, nbins, prefac, cmap = "viridis"):

    # Do an extra isolation step for specific bead counts
    beads_unique = np.unique(inputs[:, 5].astype(int))

    delta_per_bead = []
    d_sum_per_bead = []
    barrier_per_bead = []
    for bead in beads_unique:

        # Get mask for individual beads
        mask = inputs[:, 5] == str(bead)
        delta_per_bead.append(delta[mask])
        d_sum_per_bead.append(d_sum[mask])

        # Get histogram bins
        bins = np.linspace(-1, 1, nbins)
        bins_centered = (bins[1:] + bins[:-1])/2

        # Turn pooled valid deltas and sums into histogram
        data = delta[mask] / d_sum[mask]
        mirrored = np.concatenate([-data, data])
        hist, _ = np.histogram(mirrored, bins = bins)

        # Get free energy from histogram (normalized by total summed runs)
        free = - prefac * np.log(hist/hist.sum())
        finite = np.isfinite(free)

        # Shift down
        free -= np.min(free[finite])

        # Find barrier value (should be at center)
        center_idx = np.argmin(np.abs(bins_centered))
        barrier_per_bead.append(free[center_idx])

    # Tells you the value in inputs (should be repeated the whole time since these are isolated vars) corresponding to the style of that thing
    style = [(inputs[0, nametomap[key]], isostyle[key]) for key in sorted(isolated)]

    # Label from key to ordering
    keytoorder = get_keytoorder(isostyle, isolated, cmap)

    styleparams = dict([(s[1], '') for s in style])
    for i, key in enumerate(sorted(isolated)):
        styleparams[style[i][1]] = keytoorder[style[i][1]][list(map(str, isolated[key])).index(style[i][0])]

    updparams = DEF_STYLE.copy() | styleparams
    line, = ax.plot(beads_unique, barrier_per_bead, **updparams)


    ax.set_xscale('log')
    ax.set_xticks(beads_unique)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    # Label
    ax.set_xlabel(r"P")
    ax.set_ylabel(r"$\mathcal{F}(\delta=0)$ [$k_BT$]")

    return line