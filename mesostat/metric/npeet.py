import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim

import npeet.entropy_estimators as ee


# Compute metrics individually for each channel
def npeet_metric_1D_generic(method, data, settings):
    assert data.shape[settings['dim_order'].index("p")] == 1, "Expected only 1 channel for this estimate"
    methods1D = {
        'Entropy' : entropy,
        'PI'      : predictive_info
    }
    return methods1D[method](data, settings)


# Compute 1 metric for all channels
def npeet_metric_ND_generic(method, data, settings):
    methodsND = {
        'Entropy' : entropy,
        'PI'      : predictive_info
    }
    return methodsND[method](data, settings)


def entropy(data, settings):
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    return ee.entropy(dataFlat)


def _split_past_future(data, dimOrder, lag):
    test_have_dim("_split_past_future", dimOrder, "s")

    dataCanon = numpy_transpose_byorder(data, dimOrder, 'srp', augment=True)
    nTime = dataCanon.shape[0]

    x = np.array([dataCanon[i:i+lag] for i in range(nTime - lag)])

    # shape transform for x :: (swrp) -> (s*r, w*p)
    x = x.transpose((0,2,1,3))
    x = numpy_merge_dimensions(x, 2, 4)  # p*w
    x = numpy_merge_dimensions(x, 0, 2)  # s*r

    # shape transform for y :: (srp) -> (s*r, p)
    y = numpy_merge_dimensions(dataCanon[lag:], 0, 2)

    return x,y


# Predictive information
# Defined as H(Future) - H(Future | Past) = MI(Future : Past)
def predictive_info(data, settings):
    x, y = _split_past_future(data, settings['dim_order'], settings['max_lag'])
    return ee.mi(x, y)


def predictive_info_non_uniform(dataLst, settings):
    test_have_dim("_split_past_future", settings['dim_order'], "s")

    # Transpose dataset into comfortable form
    tmpDimOrder = 'ps'
    dataOrdLst = [numpy_transpose_byorder(data, settings['dim_order'], tmpDimOrder, augment=True) for data in dataLst]

    # Test that all trials have the same number of
    # Test that all trials have sufficient timesteps for lag estimation
    dataShapes = np.array([data.shape for data in dataOrdLst]).T
    assert np.all(dataShapes[0] == dataShapes[0][0]), "Number of channels must be the same for all trials"
    if np.min(dataShapes[1]) <= settings['max_lag']:
        raise ValueError('lag', settings['max_lag'], 'cannot be estimated for number of timesteps', np.min(dataShapes[1]))

    xLst = []
    yLst = []
    for dataTrial in dataOrdLst:
        x, y = _split_past_future(dataTrial, tmpDimOrder, settings['max_lag'])
        xLst += [x]
        yLst += [y]

    return ee.mi(np.vstack(xLst), np.vstack(yLst))