import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from scipy.stats import hypergeom, binom_test

from mesostat.stat.machinelearning import drop_nan_rows


#
def label_binary_data(dataA, dataB, keyA, keyB):
    nA = len(dataA)
    nB = len(dataB)
    dataTot = np.vstack([dataA, dataB])
    labelsTot = np.array([keyA]*nA + [keyB]*nB)
    return dataTot, labelsTot


def _slice_train_test(x, iStart, iEnd):
    return np.concatenate([x[:iStart], x[iEnd:]], axis=0), x[iStart:iEnd]


def kfold_iterator(x, y, k=10):
    # 1. Permute the trials to avoid local inter-trial dependencies
    nData = len(x)
    idxPerm = np.random.permutation(nData)
    xNew = x[idxPerm]
    yNew = y[idxPerm]

    delta = int(np.ceil(nData / k))
    for i in range(k):
        xTrain, xTest = _slice_train_test(xNew, i*delta, (i + 1)*delta)
        yTrain, yTest = _slice_train_test(yNew, i*delta, (i + 1)*delta)
        yield xTrain, yTrain, xTest, yTest


def leave_one_out_iterator(x, y):
    for i in range(len(x)):
        xTrain, xTest = _slice_train_test(x, i, i + 1)
        yTrain, yTest = _slice_train_test(y, i, i + 1)
        yield xTrain, yTrain, xTest, yTest


def select_cv_iterator(method, x, y, k):
    if method == "kfold":
        return kfold_iterator(x, y, k)
    elif method == "looc":
        return leave_one_out_iterator(x, y)
    else:
        raise ValueError('Unexpected method', method)


# Transform data to PCA domain, keep the largest PCAs for which the explained variance ratio sums up to a given fraction
def dim_reduction(x, expVarRatioTot):
    # Transform data to PCA domain
    pca = PCA()
    xPCA = pca.fit_transform(x)

    # Threshold components by their explained variance
    expVar = pca.explained_variance_ratio_
    expVarCDF = np.add.accumulate(expVar)
    nPCAThr = np.sum(expVarCDF < expVarRatioTot) + 1  # The borderline component that crosses the threshold should also be included
    return xPCA[:, :nPCAThr]


# Oversample the dataset, such that the number of datapoints for each class would be equal to that of the largest class
# Oversampling for each class is done by random sampling from all points of that class
def balance_oversample(x, y, classes):
    xNew = x.copy()
    yNew = y.copy()

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


def confusion_matrix(y1, y2, labels):
    nLabels = len(labels)
    rez = np.zeros((nLabels, nLabels), dtype=int)
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            rez[i, j] = np.count_nonzero((y1 == label1) & (y2 == label2))
    return rez


def weighted_accuracy(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)
    # return np.mean([cm[i,i] / np.sum(cm[i]) for i in range(len(cm))])


# Train a binary classifier to predict labels from data
# Use cross-validation to evaluate training and test accuracy
# Resample a few times to get average training and test accuracy
# TODO: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
def binary_classifier(data1, data2, classifier, method="kfold", k=10, balancing=False, pcaThr=None, havePVal=False):
    # Convert data to labeled form
    labels = [-1, 1]
    x, y = label_binary_data(data1, data2, *labels)

    # Drop NAN values
    xNoNan, yNoNan = drop_nan_rows([x, y])

    if pcaThr is not None:
        xNoNan = dim_reduction(xNoNan, pcaThr)
        print('Reduced number of dimensions to', xNoNan.shape[1])

    # map labels to binary variable
    nData = len(yNoNan)
    if nData == 0:
        print("Warning: dataset had zero non-nan rows")
        return {"acc_train": 0, "acc_test": 0, "acc_naive": 0, "p-value": 1}

    nA = np.sum(yNoNan == 1)  # Number of points with label 1
    nB = nData - nA      # Number of points with label -1

    if (nA < 2) or (nB < 2):
        print("Warning: unexpected number of labels", nA, nB, "; aborting classification")
        return 0, 0

    # Add extra dimension if X is 1D
    if xNoNan.ndim == 1:
        xNoNan = xNoNan[:, None]
        print('Warning: Got 1D data, had to add extra dimension')

    cmTrain = np.zeros((2, 2), dtype=int)
    cmTest = np.zeros((2, 2), dtype=int)

    cvfunc = select_cv_iterator(method, xNoNan, yNoNan, k)
    for xTrain, yTrain, xTest, yTest in cvfunc:
        if balancing:
            xTrainEff, yTrainEff = balance_oversample(xTrain, yTrain, labels)
        else:
            xTrainEff, yTrainEff = xTrain, yTrain

        clf = classifier.fit(xTrainEff, yTrainEff)
        # LogisticRegression(max_iter=1000)

        cmTrain += confusion_matrix(clf.predict(xTrain), yTrain, labels)
        cmTest += confusion_matrix(clf.predict(xTest), yTest, labels)

    # print('cmTrain\n', cmTrain)
    # print('cmTest\n', cmTest)

    # Accuracy
    accTrain = weighted_accuracy(cmTrain)
    accTest = weighted_accuracy(cmTest)
    rez = {"accTrain" : accTrain, "accTest" : accTest}
    if havePVal:
        rez = {**rez, **test_classifier_significance(nA, nB, cmTest)}
        # rez = {**rez, **test_classifier_significance(nA, nB, len(yTest), accTest)}

    return rez


def test_classifier_significance(nA, nB, cm):
    nTestTot = np.sum(cm)
    nTestPassed = np.sum(np.diag(cm))
    nData = nA + nB
    nBigger = np.max([nA, nB])
    pNull = nBigger / nData
    pValBinom = binom_test(nTestPassed, nTestTot, pNull, alternative='greater')
    return {"accNaive" : pNull, "pval" : pValBinom}


# FIXME: The test makes no sense. Think of a correct test
# def test_classifier_significance(nA, nB, nTest, accTest):
#     '''
#     :param nA:  Number of datapoints of first type
#     :param nB:  Number of datapoints of second type
#     :param nTest: Number of tests performed
#     :param accTest: Test accuracy obtained
#     :return: Naive accuracy, pvalue of cross-validation
#
#     https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
#     Suppose we have a collection of 20 animals, of which 7 are dogs. Then if we want to know the probability of finding a
#     given number of dogs if we choose at random 12 of the 20 animals, we can initialize a frozen distribution and plot the
#     probability mass function:
#
#     [M, n, N] = [20, 7, 12]
#     rv = hypergeom(M, n, N)
#     '''
#
#     # Calculate P-Value against following naive H0 predictor:
#     #   HO: Always guess the class which has more datapoints in the original set
#     nData = nA + nB
#     nBigger = np.max([nA, nB])             # Greatest among the two
#     # nPassTrue = int(np.ceil(accTest * nTest))  # Expected number of passed tests for real data
#     nPassTrue = int(np.ceil(accTest * nData))  # Expected number of passed tests for real data
#
#     # Probability of getting at at least as many tests passed by chance as nTrue
#     pdf = hypergeom(nData, nBigger, nTest).pmf(np.arange(0, nTest + 1))
#     # pVal = np.max([np.sum(pdf[nTrue:]), 1 / (nTest+1)])
#     pVal = np.sum(pdf[nPassTrue:])
#
#     return {"accNaive" : nBigger / nData, "pval" : pVal}