import numpy as np
from matplotlib import colors, colorbar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorsys
from scipy import interpolate

import mesostat.utils.pandas_helper as pandas_helper


def random_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def imshow(fig, ax, data, xlabel=None, ylabel=None, title=None, haveColorBar=False, limits=None, extent=None,
           xTicks=None, yTicks=None, haveTicks=False, cmap=None, aspect='auto'):
    img = ax.imshow(data, cmap=cmap, extent=extent, aspect=aspect)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if haveColorBar:
        imshowAddColorBar(fig, ax, img)
    if not haveTicks:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
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


# Add colorbar to existing imshow
def imshowAddColorBar(fig, ax, img):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')


# Adds fake colorbar to any axis. That colorbar will linearly interpolate an existing colormap
def imshowAddFakeColorBar(fig, ax, cmap, vmin=0, vmax=1):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')


def custom_grad_cmap(colorArr):
    '''
    Generate a custom colormap given colors

    :param colorArr: array of shape [nPoint, 3] - an RGB color for each interpolation point
    :return: a matplotlib CMAP. Currently distributes the interpolation points evenly
    '''

    nStep = 256
    nPoint = len(colorArr)

    colorArrEff = colorArr if np.max(colorArr) <= 1 else colorArr / 255

    x = np.arange(nStep)
    x0 = np.linspace(0, nStep - 1, nPoint)

    vals = np.ones((nStep, 4))
    vals[:, 0] = interpolate.interp1d(x0, colorArrEff[:, 0], kind='linear')(x)
    vals[:, 1] = interpolate.interp1d(x0, colorArrEff[:, 1], kind='linear')(x)
    vals[:, 2] = interpolate.interp1d(x0, colorArrEff[:, 2], kind='linear')(x)
    return ListedColormap(vals)


# Plot stacked barplot from pandas dataframe
# Xkey and Ykey specify columns that will be displayed on x and y axis
# The remaining columns will be swept over by considering all their permutations
#    * It is assumed that all the remaining columns of the dataframe are countable (e.g. not float)
def stacked_bar_plot(ax, df, xKey, yKey):
    sweepSet = set(df.columns) - {xKey, yKey}
    sweepVals = {key: list(set(df[key])) for key in sweepSet}
    sweepDF = pandas_helper.outer_product_df(sweepVals)

    bottom = np.zeros(len(set(df[xKey])))
    for idx, row in sweepDF.iterrows():
        queryDict = dict(row)
        dfThis = pandas_helper.pd_query(df, queryDict)
        ax.bar(dfThis[xKey], dfThis[yKey], bottom=bottom, label=str(list(queryDict.values())))
        bottom += np.array(dfThis[yKey])

    ax.set_xlabel(xKey)
    ax.set_ylabel(yKey)
    ax.legend()


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