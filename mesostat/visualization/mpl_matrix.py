import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import mesostat.visualization.mpl_colorbar as mpl_colorbar


def imshow(fig, ax, data, xlabel=None, ylabel=None, title=None, haveColorBar=False, limits=None, extent=None,
           xTicks=None, yTicks=None, haveTicks=False, cmap=None, aspect='auto', fontsize=20):
    img = ax.imshow(data, cmap=cmap, extent=extent, aspect=aspect)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if haveColorBar:
        mpl_colorbar.imshow_add_color_bar(fig, ax, img)
    if not haveTicks:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        # ax.axis('off')
    if limits is not None:
        norm = colors.Normalize(vmin=limits[0], vmax=limits[1])
        img.set_norm(norm)
    if xTicks is not None:
        ax.set_xticks(np.arange(len(xTicks)))
        ax.set_xticklabels(xTicks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if yTicks is not None:
        ax.set_yticks(np.arange(len(yTicks)))
        ax.set_yticklabels(yTicks)
    return img


def plot_matrix(data, shape, xlabels=None, ylabels=None, plottitles=None, lims=None, title=None, haveColorBar=False,
                xTicks=None, yTicks=None, haveTicks=False, savename=None):
    # Create plot matrix
    nRows, nCols = shape
    fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(5*nRows, 5*nCols))
    if nRows == 1:
        ax = ax[None, :]
    if nCols == 1:
        ax = ax[:, None]

    if title is not None:
        fig.suptitle(title)
    
    # Plot data
    for iRow in range(nRows):
        for iCol in range(nCols):
            iPlot = iCol + nCols*iRow
            limitsThis = lims[iPlot] if lims is not None else None
            titleThis = plottitles[iPlot] if plottitles is not None else None
            imshow(fig, ax[iRow][iCol], data[iPlot], title=titleThis, haveColorBar=haveColorBar, haveTicks=haveTicks, xTicks=xTicks, yTicks=yTicks, limits=limitsThis)

    if xlabels is not None:
        for iCol in range(nCols):
            ax[0][iCol].set_title(xlabels[iCol])

    if ylabels is not None:
        for iRow in range(nRows):
            ax[iRow][0].set_ylabel(ylabels[iRow])

    if savename is not None:
        plt.savefig(savename, dpi=300)

    return fig, ax
