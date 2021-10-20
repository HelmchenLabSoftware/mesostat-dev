import numpy as np
from matplotlib.patches import Rectangle

from mesostat.visualization.mpl_colors import base_colors_rgb


def add_timescale_bar(fig, tNames, tPosts, tNow, colorscheme='tableau', colorNow='red', fontsize=14):
    """
    :param fig:          Existing figure. May have multiple subplots (intended)
    :param tNames:       Names of time intervals
    :param tPosts:       Time at which each time interval ends
    :param tNow:         Specific current time to be marked on the timebar
    :param colorscheme:  Color scheme name from which the time interval colors will be sampled
    :param colorNow:     Color of the marker for the current time
    :param fontsize:     Font size of the timebar labels
    :return:             Nothing

    Produces a timescale bar, namely, a set of consecutive labeled rectangles denoting different time intervals.
    Attaches the timescale bar at the bottom of existing figure. Intended use is to mark the timeframe of a video, when
    multiple consecutive figures of the same shape are produced.
    """

    # Get size of the original image in inches
    wSrcInch, hSrcInch = fig.get_size_inches()

    # Postulate size of the timebar in inches
    wTrgInch = wSrcInch   # Currently, width of timebar will be equal to the full figure width
    hTrgInch = 0.3        # Currently, height of the timebar is fixed

    # Find size of the timebar in pixels
    wTrgPix = wTrgInch * fig.dpi
    hTrgPix = hTrgInch * fig.dpi

    # Find size of the timebar relative to the original image
    wRel = wTrgInch / wSrcInch
    hRel = hTrgInch / hSrcInch

    # Add a new axis just below the existing figure
    # Note: Positions and sizes must be specified relative to the original figure for this function
    # Note: timebar is shifted down exactly by its height. In theory it should overlap with original image, in practice
    #   it does not, there is some extra margin introduced by matplotlib somewhere
    ax = fig.add_axes([0, -hRel, wRel, hRel])
    _make_time_legend(ax, tNames, tPosts, wTrgPix, hTrgPix, tNow,
                      colorscheme=colorscheme, colorNow=colorNow, fontsize=fontsize)


def _make_time_legend(ax, tNames, tPosts, wPix, hPix, tNow, colorscheme='tableau', colorNow='red', fontsize=14):
    """
    :param ax:           Axis where to plot the timebar
    :param tNames:       Names of time intervals
    :param tPosts:       Time at which each time interval ends
    :param wPix:         Width of timebar in pixels
    :param hPix:         Height of timebar in pixels
    :param tNow:         Specific current time to be marked on the timebar
    :param colorscheme:  Color scheme name from which the time interval colors will be sampled
    :param colorNow:     Color of the marker for the current time
    :param fontsize:     Font size of the timebar labels
    :return:             Nothing

    Auxiliary procedure for add_timescale_bar to actually produce the timebar
    """

    nBars = len(tNames)
    epsTime2Pix = wPix / tPosts[-1]

    # Rescale timeposts from time to timebar width. Add 0-timepost at the beginning
    tPostsPix = np.hstack([[0], tPosts]) * epsTime2Pix

    # Rescale current time to timebar width
    tNowPix = tNow * epsTime2Pix

    # Get color palette to sample rectangles
    # basecolors = list(base_colors_rgb(key=colorscheme).values())
    basecolors = base_colors_rgb(key=colorscheme)

    # Plot all rectangles
    for i in range(nBars):
        bar = Rectangle((tPostsPix[i], 0), tPostsPix[i + 1] - tPostsPix[i], hPix, facecolor=basecolors[i], alpha=0.5)
        ax.add_patch(bar)

        # Label each rectangle. Each label is centered with respect to its rectangle.
        ax.text((tPostsPix[i] + tPostsPix[i + 1]) // 2, hPix // 2, tNames[i],
                fontsize=fontsize, ha='center', va='center')

    ax.axvline(x=tNowPix, color=colorNow)
    ax.set_axis_off()
    ax.autoscale()
