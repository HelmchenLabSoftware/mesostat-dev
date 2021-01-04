import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, get_list_shapes
from mesostat.metric.impl.time_splitter import split3D
from mesostat.stat.machinelearning import drop_nan_rows

import npeet.entropy_estimators as ee


def average_entropy_3D(data, settings):
    dataFlat = np.hstack(data).T  # rps -> (rs)p

    nSample, nProcess = dataFlat.shape
    if nSample < 5 + 5 * nProcess:
        # If there are too few samples, there is no point to calculate anything
        return np.array(np.nan)
    else:
        return ee.entropy(dataFlat) / nProcess


def average_tc_3D(data, settings):
    dataFlat = np.hstack(data)  # rps -> p(rs)
    dataShuffle = np.copy(dataFlat)
    for iRow in range(dataShuffle.shape[0]):
        np.random.shuffle(dataShuffle[iRow])

    return ee.kldiv(dataFlat.T, dataShuffle.T) / len(dataFlat)


# Predictive information
# Defined as H(Future) - H(Future | Past) = MI(Future : Past)
def average_predictive_info(data, settings):
    x, y = drop_nan_rows(split3D(data, settings['max_lag']))

    nSample, nProcess = x.shape
    if nSample < 5 + 5 * nProcess:
        # If there are too few samples, there is no point to calculate anything
        return np.array(np.nan)
    else:
        return ee.mi(x, y) / nProcess


def average_predictive_info_non_uniform(dataLst, settings):
    # Test that all trials have sufficient timesteps for lag estimation
    nSampleMin = np.min(get_list_shapes(dataLst, axis=1))
    if nSampleMin <= settings['max_lag']:
        raise ValueError('lag', settings['max_lag'], 'cannot be estimated for number of timesteps', nSampleMin)

    xLst = []
    yLst = []
    for dataTrial in dataLst:
        x, y = drop_nan_rows(split3D(dataTrial, settings['max_lag']))
        xLst += [x]
        yLst += [y]
    xArr = np.vstack(xLst)
    yArr = np.vstack(yLst)

    nSample, nProcess = xArr.shape
    if nSample < 4 * nProcess:
        # If there are too few samples, there is no point to calculate anything
        return np.array(np.nan)
    else:
        return ee.mi(xArr, yArr) / nProcess


# Analog of cross-correlation for pairwise mutual infos
def cross_mi_3D(data, settings):
    nTrial, nProcess, nSample = data.shape
    if nTrial*nSample < 2 * nProcess:
        # If there are too few samples, there is no point to calculate anything
        return np.full((nProcess, nProcess), np.nan)
    else:
        lag = settings['lag']

        # Check that number of timesteps is sufficient to estimate lagMax
        if nSample <= lag:
            raise ValueError('lag', lag, 'cannot be estimated for number of timesteps', nSample)

        # dataCanon = numpy_transpose_byorder(data, 'rps', 'srp')
        dataOrd = numpy_transpose_byorder(data, 'rps', 'psr')
        xx = numpy_merge_dimensions(dataOrd[:, :nSample-lag], 1, 3)
        yy = numpy_merge_dimensions(dataOrd[:, lag:], 1, 3)

        rez = np.zeros((nProcess, nProcess))
        if lag > 0:
            for i in range(nProcess):
                for j in range(nProcess):
                    rez[i][j] = ee.mi(xx[i], yy[j])
        else:
            # Optimization - take advantage of symmetry
            for i in range(nProcess):
                for j in range(i, nProcess):
                    rez[i][j] = ee.mi(xx[i], yy[j])
                    rez[j][i] = rez[i][j]

        return rez
