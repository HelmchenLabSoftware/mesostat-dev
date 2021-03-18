import numpy as np
import matplotlib.pyplot as plt

import mesostat.utils.pandas_helper as pandas_helper


def barplot_labeled(ax, data, dataLabels, plotLabel=None, alpha=None, vlines=None, hlines=None):
    dataEff = data if data.ndim == 2 else data[:, None]
    nDim, nData = dataEff.shape
    x = np.arange(nDim)
    y = np.nanmean(dataEff, axis=1)

    ax.bar(x, y, width=1, alpha=alpha, label=plotLabel)
    plt.setp(ax, xticks=x, xticklabels=dataLabels)

    if vlines is not None:
        for vline in vlines:
            ax.axvline(x=vline, color='r', linestyle='--')

    if hlines is not None:
        for hline in hlines:
            ax.axhline(y=hline, color='r', linestyle='--')


# Plot stacked barplot from pandas dataframe
# Xkey and Ykey specify columns that will be displayed on x and y axis
# The remaining columns will be swept over by considering all their permutations
#    * It is assumed that all the remaining columns of the dataframe are countable (e.g. not float)
def barplot_stacked(ax, df, xKey, yKey):
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


def barplot_stacked_indexed(ax, yDict, xTickLabels=None, title=None, xLabel=None, yLabel=None, iMax=None, rotation=45):
    bottom = np.zeros(len(next(iter(yDict.values()))))
    bottom = bottom if iMax is None else bottom[:iMax]


    for yName, yVal in yDict.items():
        yValEff = yVal if iMax is None else yVal[:iMax]
        x = np.arange(len(yValEff))

        ax.bar(x, yValEff, bottom=bottom, label=yName)
        bottom += yValEff

    ax.legend()
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
    if title is not None:
        ax.set_title(title)
    if xTickLabels is not None:
        xTickEff = xTickLabels if iMax is None else xTickLabels[:iMax]
        ax.set_xticks(np.arange(len(xTickEff)))
        ax.set_xticklabels(xTickEff)
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor")

