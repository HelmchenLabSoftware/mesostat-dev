import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim, test_uniform_dimension
from mesostat.metric.impl.time_splitter import split3D

import npeet.entropy_estimators as ee


# # Compute metrics individually for each channel
# def npeet_metric_1D_generic(method, data, settings):
#     assert data.shape[settings['dim_order'].index("p")] == 1, "Expected only 1 channel for this estimate"
#     methods1D = {
#         'Entropy' : entropy,
#         'PI'      : predictive_info
#     }
#     return methods1D[method](data, settings)
#
#
# # Compute 1 metric for all channels
# def npeet_metric_ND_generic(method, data, settings):
#     methodsND = {
#         'Entropy' : entropy,
#         'PI'      : predictive_info
#     }
#     return methodsND[method](data, settings)


def average_entropy(data, settings):
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    nSample, nProcess = dataFlat.shape
    if nSample < 2 * nProcess:
        # If there are too few samples, there is no point to calculate anything
        return np.array(np.nan)
    else:
        return ee.entropy(dataFlat) / nProcess


# Predictive information
# Defined as H(Future) - H(Future | Past) = MI(Future : Past)
def average_predictive_info(data, settings):
    x, y = split3D(data, settings['dim_order'], settings['max_lag'])
    nSample, nProcess = x.shape
    if nSample < 2 * nProcess:
        # If there are too few samples, there is no point to calculate anything
        return np.array(np.nan)
    else:
        return ee.mi(x, y) / nProcess


def average_predictive_info_non_uniform(dataLst, settings):
    test_have_dim("_split_past_future", settings['dim_order'], "s")
    test_uniform_dimension(dataLst, settings['dim_order'], "p")
    # Test that all trials have sufficient timesteps for lag estimation
    idxSample = settings['dim_order'].index('s')
    nSampleMin = np.min([data.shape[idxSample] for data in dataLst])
    if nSampleMin <= settings['max_lag']:
        raise ValueError('lag', settings['max_lag'], 'cannot be estimated for number of timesteps', nSampleMin)

    # # Transpose dataset into comfortable form
    # tmpDimOrder = 'ps'
    # dataOrdLst = [numpy_transpose_byorder(data, settings['dim_order'], tmpDimOrder, augment=True) for data in dataLst]

    xLst = []
    yLst = []
    for dataTrial in dataLst:
        x, y = split3D(dataTrial, settings['dim_order'], settings['max_lag'])
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