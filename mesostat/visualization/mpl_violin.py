import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from IPython.display import display
from statannot import add_stat_annotation

from mesostat.stat.testing.htests import mannwhitneyu_nan_aware


def violins_labeled(ax, dataLst, dataLabels, paramName, metricName, joinMeans=True, haveLog=False, sigTestPairs=None, printLogP=False, violinInner=None):
    '''
    :param ax: plot axis
    :param dataLst: 1D datasets for each violin. Can be different length
    :param dataLabels: Labels for each dataset
    :param paramName: label of the x-axis
    :param metricName: label of the y-axis
    :param joinMeans: if True, joins the means of consecutive violins
    :param haveLog: if True, plot is in log-scale
    :param sigTestPairs: a set of pairs of indices of datasets for which the statistical annotation must be drawn
    :param printLogP: if True, print the p-value of MannWhitneyU test for every pair of datasets
    :param violinInner: the style of the inner block of the violinplot
    '''

    data = np.concatenate(dataLst)
    nDataLst = [len(d) for d in dataLst]

    labels = np.repeat(dataLabels, nDataLst)
    df = pd.DataFrame({paramName: labels, metricName: data})

    if haveLog:
        ax.set_yscale("log")

    #####################
    # Plot violins
    #####################
    sns.violinplot(ax=ax, data=df, x=paramName, y=metricName, cut=0, inner=violinInner)

    #####################
    # Plot annotation
    #####################
    if sigTestPairs is not None:
        labelPairs = []
        for i, j in sigTestPairs:
            labelPairs += [(dataLabels[i], dataLabels[j])]

            print("For", labelPairs[-1],
                  "of data size", (len(dataLst[i]), len(dataLst[j])),
                  "rank-sum-test is", mannwhitneyu(dataLst[i], dataLst[j], alternative='two-sided')[1])

        add_stat_annotation(ax, data=df, x=paramName, y=metricName, box_pairs=labelPairs, test='Mann-Whitney', loc='inside', verbose=0)  # , text_format='full'

    #####################
    # Join Means
    #####################
    tickCoords = np.arange(len(dataLabels))
    # plt.setp(ax, xticks=tickCoords, xticklabels=dataLabels)
    if joinMeans:
        ax.plot(tickCoords, [np.nanmean(d) for d in dataLst], '*--', color='blue')

    #####################
    # Print p-value matrix
    #####################
    if printLogP:
        nData = len(dataLst)
        logPValArr = np.zeros((nData, nData))
        for iData in range(nData):
            for jData in range(iData + 1, nData):
                logPValArr[iData][jData] = np.round(mannwhitneyu_nan_aware(dataLst[iData], dataLst[jData])[0], 2)

        logPValArr = logPValArr + logPValArr.T
        display(logPValArr)