import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from scipy.stats import hypergeom

from mesostat.stat.stat import rand_bool_perm

def _get_nan_rows(x):
    if not np.issubdtype(x.dtype, np.number):
        return np.zeros(x.shape).astype(bool)
    elif x.ndim <= 1:
        return np.isnan(x)
    else:
        return np.any(np.isnan(x), axis=tuple(list(range(1, x.ndim))))

# Drop all rows for which there is at least one NAN value in X or Y
def drop_nan_rows(x, y):
    nanX = _get_nan_rows(x)
    nanY = _get_nan_rows(y)
    goodIdx = (~nanX) & (~nanY)
    return x[goodIdx], y[goodIdx]


def _slice_train_test(x, iStart, iEnd):
    return np.concatenate([x[:iStart], x[iEnd:]], axis=0), x[iStart:iEnd]


# CrossValidation: Split data into training and test sets along the first dimension
def split_train_test(dataLst, fracTest):
    n = len(dataLst[0])
    nTest = int(fracTest * n)
    testIdxs = rand_bool_perm(nTest, n)
    trainLst = [data[~testIdxs] for data in dataLst]
    testLst = [data[testIdxs] for data in dataLst]
    return trainLst + testLst


def kfold_iterator(x, y, k=10):
    # 1. Permute the data just in case, to avoid very non-random initializations
    nData = len(x)
    idxPerm = np.random.permutation(nData)
    xNew = x[idxPerm]
    yNew = y[idxPerm]

    delta = int(np.ceil(nData / k))
    for i in range(k):
        xTrain, xTest = _slice_train_test(x, i*delta, (i + 1)*delta)
        yTrain, yTest = _slice_train_test(y, i*delta, (i + 1)*delta)
        yield xTrain, yTrain, xTest, yTest


def leave_one_out_iterator(x, y):
    for i in range(len(x)):
        xTrain, xTest = _slice_train_test(x, i, i + 1)
        yTrain, yTest = _slice_train_test(y, i, i + 1)
        yield xTrain, yTrain, xTest, yTest


# Oversample the dataset, such that the number of datapoints for each class would be equal to that of the largest class
# Oversampling for each class is done by random sampling from all points of that class
def balance_oversample(x, y):
    xNew = x.copy()
    yNew = y.copy()

    classes = set(y)
    nDataPerClass = [np.count_nonzero(y == label) for label in classes]
    nDataMax = np.max(nDataPerClass)
    nExtraDataPerClass = [nDataMax - nData for nData in nDataPerClass]

    for label, nData, nExtraData in zip(classes, nDataPerClass, nExtraDataPerClass):
        if nExtraData > 0:
            idxsThis = y == label
            idxsResample = np.random.randint(0, nData, nExtraData)
            xNew = np.concatenate((xNew, x[idxsThis][idxsResample]), axis=0)
            yNew = np.concatenate((yNew, y[idxsThis][idxsResample]), axis=0)

    return xNew, yNew


def confusion_matrix(y1, y2):
    labels = sorted(set(y1) | set(y2))
    nLabels = len(labels)
    rez = np.zeros((nLabels, nLabels), dtype=int)
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            rez[i, j] = np.count_nonzero((y1 == label1) & (y2 == label2))
    return rez


def weighted_accuracy(cm):
    return np.mean([cm[i,i] / np.sum(cm[i]) for i in range(len(cm))])


# https://en.wikipedia.org/wiki/Cohen%27s_kappa
def cohen_kappa(cm):
    cmNorm = cm / np.sum(cm)
    p0 = np.sum(cmNorm.diag())
    pe = np.sum([np.sum(cmNorm[i]) * np.sum(cmNorm[:, i]) for i in range(len(cm))])
    return (p0 - pe) / (1 - pe)


# Train a binary classifier to predict labels from data
# Use cross-validation to evaluate training and test accuracy
# Resample a few times to get average training and test accuracy
# TODO: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
def binary_classifier(x, y, method="kfold", balancing=False):
    # Test consistency
    assert len(x) == len(y), "Labels and values must match"
    assert len(set(y)) == 2, "Labels must be binary"

    # Drop NAN values
    xNoNan, yNoNan = drop_nan_rows(x, y)

    # map labels to binary variable
    nData = len(yNoNan)
    if nData == 0:
        print("Warning: dataset had zero non-nan rows")
        return {"acc_train": 0, "acc_test": 0, "acc_naive": 0, "p-value": 1}

    # Add extra dimension if X is 1D
    if xNoNan.ndim == 1:
        xNoNan = xNoNan[:, None]

    labels = set(yNoNan)
    nLabels = len(labels)
    cmTrain = np.zeros((nLabels, nLabels), dtype=int)
    cmTest = np.zeros((nLabels, nLabels), dtype=int)
    yBinary = yNoNan == yNoNan[0]
    nA = np.sum(yBinary)                    # Number of points with label 1
    nB = nData - nA                         # Number of points with label 0

    if (nLabels != 2) or (nA < 2) or (nB < 2):
        print("Warning: Effective number of labels is", nLabels, "; aborting classification")
        return {"acc_train": 0, "acc_test": 0, "acc_naive": 0, "p-value": 1}

    cvfunc = kfold_iterator(x, y) if method=="kfold" else leave_one_out_iterator(xNoNan, yNoNan)
    for xTrain, yTrain, xTest, yTest in cvfunc:
        if balancing:
            xTrainEff, yTrainEff = balance_oversample(xTrain, yTrain)
        else:
            xTrainEff, yTrainEff = xTrain, yTrain

        clf = LogisticRegression(max_iter=1000).fit(xTrainEff, yTrainEff)

        cmTrain += confusion_matrix(clf.predict(xTrain), yTrain)
        cmTest += confusion_matrix(clf.predict(xTest), yTest)

    # Accuracy
    accTrain = weighted_accuracy(cmTrain)
    accTest = weighted_accuracy(cmTest)

    '''
    >>> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
    Suppose we have a collection of 20 animals, of which 7 are dogs. Then if we want to know the probability of finding a
    given number of dogs if we choose at random 12 of the 20 animals, we can initialize a frozen distribution and plot the
    probability mass function:
    
    [M, n, N] = [20, 7, 12]
    rv = hypergeom(M, n, N)
    '''

    # Calculate P-Value against following naive H0 predictor:
    #   HO: Always guess the class which has more datapoints in the original set
    nBigger = np.max([nA, nB])              # Greatest among the two
    nTest = len(xTest)                      # Number of tests performed
    nTrue = int(np.ceil(accTest * nTest))   # Expected number of passed tests for real data

    # Probability of getting at at least as many tests passed by chance as nTrue
    pdf = hypergeom(nData, nBigger, nTest).pmf(np.arange(0, nTest + 1))
    #pVal = np.max([np.sum(pdf[nTrue:]), 1 / (nTest+1)])
    pVal = np.sum(pdf[nTrue:])

    return {"acc_train" : accTrain, "acc_test" : accTest, "acc_naive" : nBigger / nData, "p-value" : pVal}
