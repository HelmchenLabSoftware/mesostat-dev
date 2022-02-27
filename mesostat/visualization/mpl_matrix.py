import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt

import mesostat.visualization.mpl_colorbar as mpl_colorbar
from mesostat.utils.pandas_helper import outer_product_df, pd_is_one_row, pd_query


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


def plot_df_2D_outer_product(ax, df, colNamesRow, colNamesCol, colNameTrg, vmin=None, vmax=None, orderDict=None, dropEmpty=False):

    def _drop_empty_queries(df, dfQueries):
        idxs = [idx for idx, row in dfQueries.iterrows() if len(pd_query(df, dict(row))) == 0]
        return dfQueries.drop(idxs, axis=0).reset_index(drop=True)

    dictRow = {colName: sorted(list(set(list(df[colName])))) for colName in colNamesRow}
    dictCol = {colName: sorted(list(set(list(df[colName])))) for colName in colNamesCol}

    if orderDict is not None:
        for k, v in orderDict.items():
            if k in dictRow.keys():
                dictRow[k] = v
            if k in dictCol.keys():
                dictCol[k] = v

    dfOuterRow = outer_product_df(dictRow)
    dfOuterCol = outer_product_df(dictCol)

    if dropEmpty:
        dfOuterRow = _drop_empty_queries(df, dfOuterRow)
        dfOuterCol = _drop_empty_queries(df, dfOuterCol)

    colLabelsRow = ['_'.join(list(row)) for idx, row in dfOuterRow.iterrows()]
    colLabelsCol = ['_'.join(list(row)) for idx, row in dfOuterCol.iterrows()]

    # Assemble 2D matrix of all results
    rezArr = np.zeros((len(colLabelsRow), len(colLabelsCol)))
    for idxRow, rowRow in dfOuterRow.iterrows():
        for idxCol, rowCol in dfOuterCol.iterrows():
            # Join both queries
            queryDict = {**dict(rowRow), **dict(rowCol)}

            # Do query
            dfQuery = pd_query(df, queryDict)
            if len(dfQuery) == 0:
                val = np.nan
            elif len(dfQuery) == 1:
                val = pd_is_one_row(dfQuery)[1][colNameTrg]
            else:
                raise ValueError('Ups')

            rezArr[idxRow, idxCol] = val

    rezDF = pd.DataFrame(rezArr, index=colLabelsRow, columns=colLabelsCol)

    sns.heatmap(rezDF, ax=ax, vmin=vmin, vmax=vmax, cmap='jet')
    ax.set_xlabel('')
    ax.set_ylabel('')
