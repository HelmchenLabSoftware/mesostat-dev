import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from scipy.stats import hypergeom

from mesostat.stat.stat import rand_bool_perm


# CrossValidation: Split data into training and test sets along the first dimension
def split_train_test(dataLst, fracTest):
    n = len(dataLst[0])
    nTest = int(fracTest * n)
    testIdxs = rand_bool_perm(nTest, n)
    trainLst = [data[~testIdxs] for data in dataLst]
    testLst = [data[testIdxs] for data in dataLst]
    return trainLst + testLst


def _get_nan_rows(x):
    if x.ndim <= 1:
        return np.isnan(x)
    else:
        return np.any(np.isnan(x), axis=tuple(list(range(1, x.ndim))))

# Drop all rows for which there is at least one NAN value in X or Y
def drop_nan_rows(x, y):
    nanX = _get_nan_rows(x)
    nanY = _get_nan_rows(y)
    goodIdx = (~nanX) & (~nanY)
    return x[goodIdx], y[goodIdx]


# Train a binary classifier to predict labels from data
# Use cross-validation to evaluate training and test accuracy
# Resample a few times to get average training and test accuracy
def binary_classifier(x, y, nShuffle=100, pTest=0.1):
    # map Y to binary
    binaryMap = {k: i for i, k in enumerate(set(y))}
    yBinary = np.array([binaryMap[yElem] for yElem in y])

    # Test consistency
    assert len(x) == len(y)
    assert len(binaryMap) == 2

    accTrainLst = []
    accTestLst = []
    for i in range(nShuffle):
        xTrain, yTrain, xTest, yTest = split_train_test([x, yBinary], pTest)
        clf = LogisticRegression(max_iter=1000).fit(xTrain, yTrain)
        accTrainLst += [clf.score(xTrain, yTrain)]
        accTestLst += [clf.score(xTest, yTest)]

    # Accuracy
    accTrain = np.mean(accTrainLst)
    accTest = np.mean(accTestLst)

    # Calculate P-Value against following H0:
    #    Always guess the class which has more datapoints in the original set
    nA = np.sum(yBinary)
    nB = len(yBinary) - nA
    nBigger = np.max([nA, nB])
    nSmaller = np.min([nA, nB])
    nTest = len(xTest)
    nTrue = int(np.ceil(accTest * nTest))  # Expected number of passed tests for real data

    pdf = hypergeom(nBigger + nSmaller, nTest, nBigger).pmf(np.arange(0, nTest + 1))
    pVal = np.sum(pdf[nTrue:])  # Probability of getting at at least as many tests passed by chance as nTrue

    return {"Accuracy_Train" : accTrain, "Accuracy_Test : " : accTest, "p-value" : pVal}
