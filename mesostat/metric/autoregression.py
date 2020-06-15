import numpy as np

from mesostat.utils.signals import zscore, zscore_list
from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim, get_uniform_dim_shape
import mesostat.metric.impl.ar as ar
import mesostat.metric.impl.mar as mar
import mesostat.metric.impl.mar_inp as mar_inp
import mesostat.metric.impl.time_splitter as splitter
from mesostat.stat.machinelearning import split_train_test, drop_nan_rows

# FIXME: Decide if we need to ZSCORE at all or not for this metric. If yes, make MAR and AR consistent



def _preprocess_ar(data, settings):
    # Flatten processes and repetitions, zscore
    dataFlat = numpy_merge_dimensions(data, 0, 2)
    return zscore(dataFlat)


def _preprocess_ar_non_uniform(dataLst, settings):
    # Flatten processes and repetitions
    rez = []
    for data2D in dataLst:
        rez += list(data2D)

    # ZScore result
    return zscore_list(rez)



# TODO: Unstack Y in order to separate time and trial dimensions
# def mar_unstack_result(y, nTrial):
#     nTimeTrialReduced, nChannel = y.shape


def _ar_2D_alpha(x, y):
    if len(x) < 10:
        raise ValueError("Number of samples", len(x), "too small for autoregression")

    return ar.fit(x, y)[0]  # Return a scalar, not 0D array


# TODO: Perform exhaustive cross-validation
def _ar_2D_testerr(x, y, testFrac):
    xTrain, yTrain, xTest, yTest = split_train_test([x, y], testFrac)
    alpha = ar.fit(xTrain, yTrain)

    if len(xTest) < 10:
        raise ValueError("Number of samples", len(x), "too small for autoregression")

    #yHatTrain = ar.predict(xTrain, alpha)
    #trainErr = ar.rel_err(yTrain, yHatTrain)

    yHatTest = ar.predict(xTest, alpha)
    return ar.rel_err(yTest, yHatTest)


def _mar3D_alpha(x, y):
    if len(x) < 10:
        raise ValueError("Number of samples", len(x), "too small for autoregression")

    return mar.fit_mle(x, y)  # Note this is an array


def _mar3D_testerr(x, y, testFrac):
    xTrain, yTrain, xTest, yTest = split_train_test([x, y], testFrac)
    alpha = mar.fit_mle(xTrain, yTrain)  # Note this is an array

    if len(xTest) < 10:
        raise ValueError("Number of samples", len(x), "too small for autoregression")

    #yHatTrain = mar.predict(xTrain, alpha)
    #trainErr = mar.rel_err(yTrain, yHatTrain)

    yHatTest = mar.predict(xTest, alpha)
    return mar.rel_err(yTest, yHatTest)


def _preprocess_mar_inp(data, inp, nHist):
    x, y = splitter.split3D(data, nHist)

    assert inp.ndim == 3, "Input matrix must be a 3D matrix"
    assert np.prod(inp.shape) != 0, "Input matrix is degenerate"
    nTr, nCh, nT = data.shape
    nTrInp, nChInp, nTInp = inp.shape
    assert nTr == nTrInp, "Input shape must be consistent with data shape"
    assert nT == nTInp, "Input shape must be consistent with data shape"

    # Convert input into the form (rps) -> (r*s, p)
    inpCanon = numpy_transpose_byorder(inp, 'rps', 'rsp')
    u = numpy_merge_dimensions(inpCanon[:, nHist:], 0, 2)

    # Drop any nan rows that are present in the data or input
    return drop_nan_rows([x, y, u])


def _preprocess_mar_inp_non_uniform(dataLst, inpLst, nHist):
    x, y = splitter.split3D_non_uniform(dataLst, nHist)


    assert len(dataLst) == len(inpLst), "Input must have same number of trials as data"
    for data, inp in zip(dataLst, inpLst):
        assert inp.ndim == 2, "Input must be a list of 2D matrices"
        assert inp.shape[1] == data.shape[1], "Input must have same number of timesteps as data"

    # Test that input has the same number of features for each trial
    nChInp = get_uniform_dim_shape(inpLst, axis=1)

    # shape transform for y :: (rps) -> (r*s, p)
    u = [inp[:, nHist:].T for inp in inpLst]   # (rps) -> (rsp)
    u = np.concatenate(u, axis=0)              # (rsp) -> (r*s, p)

    # Drop any nan rows that are present in the data or input
    return drop_nan_rows([x, y, u])


def _mar3D_inp_testerr(x, y, u, testFrac):
    xTrain, yTrain, uTrain, xTest, yTest, uTest = split_train_test([x, y, u], testFrac)
    alpha, beta = mar_inp.fit_mle(xTrain, yTrain, uTrain)  # Note this is an array

    if len(xTest) < 10:
        raise ValueError("Number of samples", len(x), "too small for autoregression")

    #yHatTrain = mar_inp.predict(xTrain, alpha, uTrain, beta)
    #trainErr = mar_inp.rel_err(yTrain, yHatTrain)

    yHatTest = mar_inp.predict(xTest, alpha, uTest, beta)
    testErr = mar_inp.rel_err(yTest, yHatTest)
    return alpha, beta, testErr


################################################
#  Uniform Estimators
################################################


# Compute the coefficient of the AR(1) process
def ar1_coeff(data, settings):
    data2D = _preprocess_ar(data, settings)
    x, y = drop_nan_rows(splitter.split2D(data2D, 1))
    return _ar_2D_alpha(x, y)


# Compute the relative fitness error of the AR(1) process
def ar1_testerr(data, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1
    data2D = _preprocess_ar(data, settings)
    x, y = drop_nan_rows(splitter.split2D(data2D, 1))
    return _ar_2D_testerr(x, y, testFrac)


# Compute the relative fitness error of the AR(1) process
def mar1_coeff(data, settings):
    x, y = drop_nan_rows(splitter.split3D(data, 1))
    return _mar3D_alpha(x, y)


# Compute the relative fitness error of the AR(1) process
def mar1_testerr(data, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1
    x, y = drop_nan_rows(splitter.split3D(data, 1))
    return _mar3D_testerr(x, y, testFrac)


# Compute the relative fitness error of the AR(n) process for a few small n values
def ar_testerr(data, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1

    data2D = _preprocess_ar(data, settings)
    x, y = drop_nan_rows(splitter.split2D(data2D, settings['hist']))
    return _ar_2D_testerr(x, y, testFrac)


# Compute the relative fitness error of the MAR(n) process for a few small n values
def mar_testerr(data, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1

    x, y = drop_nan_rows(splitter.split3D(data, settings['hist']))
    return _mar3D_testerr(x, y, testFrac)


# Compute the relative fitness error of the MAR_INP(n) process for a few small n values
def mar_inp_testerr(data, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1

    x, y, u = _preprocess_mar_inp(data, settings['inp'], settings['hist'])
    return _mar3D_inp_testerr(x, y, u, testFrac)


################################################
#  Non-Uniform Estimators
################################################

# Compute the coefficient of the AR(1) process
def ar1_coeff_non_uniform(dataLst3D, settings):
    dataLst2D = _preprocess_ar_non_uniform(dataLst3D, settings)
    x, y = drop_nan_rows(splitter.split2D_non_unifrom(dataLst2D, 1))
    return _ar_2D_alpha(x, y)


# Compute the relative fitness error of the AR(1) process
def ar1_testerr_non_uniform(dataLst3D, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1
    dataLst2D = _preprocess_ar_non_uniform(dataLst3D, settings)
    x, y = drop_nan_rows(splitter.split2D_non_unifrom(dataLst2D, 1))
    return _ar_2D_testerr(x, y, testFrac)


# Compute the relative fitness error of the AR(1) process
def mar1_coeff_non_uniform(dataLst, settings):
    x, y = drop_nan_rows(splitter.split3D_non_uniform(dataLst, 1))
    return _mar3D_alpha(x, y)


# Compute the relative fitness error of the AR(1) process
def mar1_testerr_non_uniform(dataLst, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1
    x, y = drop_nan_rows(splitter.split3D_non_uniform(dataLst, 1))
    return _mar3D_testerr(x, y, testFrac)


# Compute the relative fitness error of the AR(n) process for a few small n values
def ar_testerr_non_uniform(dataLst, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1

    dataLst2D = _preprocess_ar_non_uniform(dataLst, settings)
    x, y = drop_nan_rows(splitter.split2D_non_unifrom(dataLst2D, settings['hist']))
    return _ar_2D_testerr(x, y, testFrac)


# Compute the relative fitness error of the MAR(n) process for a few small n values
def mar_testerr_non_uniform(dataLst, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1

    x, y = drop_nan_rows(splitter.split3D_non_uniform(dataLst, settings['hist']))
    return _mar3D_testerr(x, y, testFrac)


# Compute the relative fitness error of the MAR_INP(n) process for a few small n values
def mar_inp_testerr_non_uniform(dataLst, settings):
    testFrac = settings['testfrac'] if 'testfrac' in settings.keys() else 0.1

    x, y, u = _preprocess_mar_inp_non_uniform(dataLst, settings['inp'], settings['hist'])
    return _mar3D_inp_testerr(x, y, u, testFrac)
