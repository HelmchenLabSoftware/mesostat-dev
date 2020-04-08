from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imshow(fig, ax, data, xlabel=None, ylabel=None, title=None, haveColorBar=False, limits=None, haveTicks=False):
    img = ax.imshow(data)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if haveColorBar:
        imshowAddColorBar(fig, ax, img)
    if not haveTicks:
        ax.set_axis_off()
    if limits is not None:
        norm = colors.Normalize(vmin=limits[0], vmax=limits[1])
        img.set_norm(norm)
    return img


def plot_matrix(data, shape, xlabels=None, ylabels=None, plottitles=None, lims=None, title=None, haveColorBar=False, haveTicks=False, savename=None):
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
            imshow(fig, ax[iRow][iCol], data[iPlot], title=titleThis, haveColorBar=haveColorBar, haveTicks=haveTicks, limits=limitsThis)

    if xlabels is not None:
        for iCol in range(nCols):
            ax[0][iCol].set_title(xlabels[iCol])

    if ylabels is not None:
        for iRow in range(nRows):
            ax[iRow][0].set_ylabel(ylabels[iRow])

    if savename is not None:
        plt.savefig(savename, dpi=300)

    return fig, ax


# Add colorbar to existing imshow
def imshowAddColorBar(fig, ax, img):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')