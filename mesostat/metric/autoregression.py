import numpy as np

from mesostat.utils.signals import zscore
from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
import mesostat.metric.impl.ar as ar
import mesostat.metric.impl.mar as mar
import mesostat.metric.impl.mar_inp as mar_inp
from mesostat.metric.impl.time_splitter import split2D, split3D
from mesostat.stat.machinelearning import split_train_test


def _preprocess_ar(data, settings):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    return zscore(dataFlat)


# TODO: Unstack Y in order to separate time and trial dimensions
def mar_unstack_result(y, nTrial):
    nTimeTrialReduced, nChannel = y.shape


def _ar_2D(data2D, nHist, testFrac):
    x, y = split2D(data2D, nHist)
    xTrain, yTrain, xTest, yTest = split_train_test([x, y], testFrac)
    alpha = ar.fit(xTrain, yTrain)[0]  # Return a scalar, not 0D array

    #yHatTrain = ar.predict(xTrain, alpha)
    #trainErr = ar.rel_err(yTrain, yHatTrain)

    yHatTest = ar.predict(xTest, alpha)
    testErr = ar.rel_err(yTest, yHatTest)
    return alpha, testErr


def _mar3D(data, settings, nHist, testFrac):
    x, y = split3D(data, settings['dim_order'], nHist)
    xTrain, yTrain, xTest, yTest = split_train_test([x, y], testFrac)
    alpha = mar.fit_mle(xTrain, yTrain)  # Note this is an array

    #yHatTrain = mar.predict(xTrain, alpha)
    #trainErr = mar.rel_err(yTrain, yHatTrain)

    yHatTest = mar.predict(xTest, alpha)
    testErr = mar.rel_err(yTest, yHatTest)
    return alpha, testErr


def _mar3D_inp(data, inp, settings, nHist, testFrac):
    x, y = split3D(data, settings['dim_order'], nHist)

    # FIXME: ensure dimension order consistent with solver
    inpCanon = numpy_transpose_byorder(inp, settings['dim_order'], 'srp', augment=True)
    u = numpy_merge_dimensions(inpCanon[nHist:], 0, 2)

    xTrain, yTrain, uTrain, xTest, yTest, uTest = split_train_test([x, y, u], testFrac)
    alpha, beta = mar_inp.fit_mle(xTrain, yTrain, uTrain)  # Note this is an array

    #yHatTrain = mar_inp.predict(xTrain, alpha, uTrain, beta)
    #trainErr = mar_inp.rel_err(yTrain, yHatTrain)

    yHatTest = mar_inp.predict(xTest, alpha, uTest, beta)
    testErr = mar_inp.rel_err(yTest, yHatTest)
    return alpha, beta, testErr


# Compute the coefficient of the AR(1) process
def ar1_coeff(data, settings, testFrac=0.1):
    data2D = _preprocess_ar(data, settings)
    alpha, testErr = _ar_2D(data2D, 1, testFrac)
    return alpha


# Compute the relative fitness error of the AR(1) process
def ar1_testerr(data, settings, testFrac=0.1):
    data2D = _preprocess_ar(data, settings)
    alpha, testErr = _ar_2D(data2D, 1, testFrac)
    return testErr


# Compute the relative fitness error of the AR(1) process
def mar1_testerr(data, settings, testFrac=0.1):
    alpha, testErr = _mar3D(data, settings, 1, testFrac)
    return testErr


# Compute the relative fitness error of the AR(n) process for a few small n values
def ar_testerr(data, settings, testFrac=0.1, maxHist=10):
    data2D = _preprocess_ar(data, settings)

    testErrLst = []
    for iHist in range(1, maxHist+1):
        alpha, testErr = _ar_2D(data2D, iHist, testFrac)
        testErrLst += [testErr]
    return np.array(testErrLst)


# Compute the relative fitness error of the AR(n) process for a few small n values
def mar_testerr(data, settings, testFrac=0.1, maxHist=10):
    testErrLst = []
    for iHist in range(1, maxHist+1):
        alpha, testErr = _mar3D(data, settings, iHist, testFrac)
        testErrLst += [testErr]
    return np.array(testErrLst)