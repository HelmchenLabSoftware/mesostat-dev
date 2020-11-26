import numpy as np
import pandas as pd
from scipy.stats import f as f_distr

from mesostat.utils.pandas_helper import outer_product_df


def as_pandas(arr, colNames, valName='rez'):
    colIdxDict = {colName : np.arange(colSize) for colName, colSize in zip(colNames, arr.shape)}
    df = outer_product_df(colIdxDict)
    df[valName] = arr.flatten()
    return df


def as_pandas_lst(arrLst, colNames, arrNames, arrCondName, valName='rez'):
    dfRez = pd.DataFrame()
    for arr, arrName in zip(arrLst, arrNames):
        dfThis = as_pandas(arr, colNames, valName=valName)
        dfThis[arrCondName] = arrName
        dfRez = dfRez.append(dfThis)
    return dfRez


def anova_homebrew(dataDF, keyrez):
    muTot = np.mean(dataDF[keyrez])
    SST = np.sum((dataDF[keyrez] - muTot) ** 2)
    nT = len(dataDF)
    rezLst = [("tot", nT, SST)]

    getRow = lambda colName, colVal: dataDF[dataDF[colName] == colVal][keyrez]

    colNames = set(dataDF.columns) - {keyrez}
    for colName in colNames:
        colVals = set(dataDF[colName])

        muB = [np.mean(getRow(colName, colVal)) for colVal in colVals]
        nB = len(colVals)
        prefix = len(getRow(colName, list(colVals)[0]))
        SSB = np.sum((muB - muTot) ** 2) * prefix

        rezLst += [(colName, nB, SSB)]

    nE = 0
    SSE = rezLst[0][2] - np.sum([r[2] for r in rezLst[1:]])
    rezLst += [('err', nE, SSE)]

    rezDF = pd.DataFrame(rezLst, columns=('axis', 'nDim', 'sumsq'))

    # # Calculating degrees of freedom
    rezDF['df'] = rezDF['nDim'] - 1
    rezDF.at[len(rezDF)-1, 'df'] = 0
    rezDF.at[len(rezDF)-1, 'df'] = 2*rezDF.at[0, 'df'] - np.sum(rezDF['df'])
    del rezDF['nDim']  # This is a proxy column to calculate df, not informative by itself

    # Calculating mean square error and F-ratio
    rezDF['meansq'] = rezDF['sumsq'] / rezDF['df']
    rezDF['F'] = rezDF['meansq'] / list(rezDF[rezDF['axis'] == 'err']['meansq'])[0]
    df2 = rezDF.at[len(rezDF)-1, 'df']
    rezDF['pval'] = [1 - f_distr.cdf(f, df1, df2) for f, df1 in zip(rezDF['F'], rezDF['df'])]
    return rezDF