import numpy as np

from mesostat.utils.signals import zscore
from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
from mesostat.metric.impl.autoregression1D import AR1D
from mesostat.stat.machinelearning import split_train_test


# Compute the coefficient of the AR(1) process
def ar1_coeff(data, settings, testFrac=0.1):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    dataThis = zscore(dataFlat)

    ar1D = AR1D(nHist=1)
    x, y = ar1D.data2xy(dataThis)
    xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
    return ar1D.fit(xTrain, yTrain)[0]  # Return a scalar, not 0D array


# Compute the relative fitness error of the AR(1) process
def ar1_testerr(data, settings, testFrac=0.1):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    dataThis = zscore(dataFlat)

    ar1D = AR1D(nHist=1)
    x, y = ar1D.data2xy(dataThis)
    xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
    ar1D.fit(xTrain, yTrain)

    #yHatTrain = ar1D.predict(xTrain)
    #trainErr = ar1D.rel_err(yTrain, yHatTrain)

    yHatTest = ar1D.predict(xTest)
    testErr = ar1D.rel_err(yTest, yHatTest)
    return testErr


# Compute the relative fitness error of the AR(n) process for a few small n values
def ar_testerr(data, settings, testFrac=0.1, maxHist=10):
    test_have_dim("ar1_coeff", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    dataThis = zscore(dataFlat)

    testErrLst = []
    for iHist in range(1, maxHist+1):
        ar1D = AR1D(nHist=iHist)
        x, y = ar1D.data2xy(dataThis)
        xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
        ar1D.fit(xTrain, yTrain)

        #yHatTrain = ar1D.predict(xTrain)
        #trainErr = ar1D.rel_err(yTrain, yHatTrain)

        yHatTest = ar1D.predict(xTest)
        testErrLst += [ar1D.rel_err(yTest, yHatTest)]
    return np.array(testErrLst)
