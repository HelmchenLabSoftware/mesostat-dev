import numpy as np
import matplotlib.pyplot as plt


# Convert pvalue to stars as text
def pval_to_stars(pVal, nStarsMax=4):
    if pVal > 0.05:
        return 'NS'
    else:
        nStars = int(-np.log10(pVal))
        nStars = np.min([nStars, nStarsMax])
        return '*' * nStars


# Place significance stars between two patches, such as barplot bars
def stat_annot_patches(ax, p1, p2, pVal, **kwargs):
    x1 = p1.get_x() + p1.get_width() / 2
    x2 = p2.get_x() + p2.get_width() / 2
    y1 = p1.get_y() + p1.get_height()
    y2 = p2.get_y() + p2.get_height()
    stat_annot_coords(ax, x1, x2, y1, y2, pVal, **kwargs)


# Place significance stars on top of two objects of given height (such as bars)
def stat_annot_coords(ax, x1, x2, y1, y2, pVal, dy=0.05, dh=0.03, barh=0.03, fontsize=None):
    # Adjust plot height to allow space for stars
    y1ax, y2ax = ax.get_ylim()
    y2ax += (y2ax - y1ax) * dy
    ax.set_ylim([y1ax, y2ax])

    # Calculate relative sizes of the bar
    dh *= (y2ax - y1ax)
    barh *= (y2ax - y1ax)

    # Calculate bar coordinates and plot it
    ybar = max(y1, y2) + dh
    barx = [x1, x1, x2, x2]
    bary = [ybar, ybar + barh, ybar + barh, ybar]
    mid = ((x1 + x2) / 2, ybar + barh)
    ax.plot(barx, bary, c='black')

    # Adjust text properties
    kwargs = dict(ha='center', va='bottom')
    if fontsize is not None:
        kwargs['fontsize'] = fontsize

    # Find the number of stars and plot them
    pValText = pval_to_stars(pVal)
    plt.text(*mid, pValText, **kwargs)