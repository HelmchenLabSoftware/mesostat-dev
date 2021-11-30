import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.stats import mannwhitneyu, wilcoxon, binom

rstest_twosided = lambda x, y : mannwhitneyu(x, y, alternative='two-sided')


# Computes 1D empirical tolerance interval by determining (1 - sigma)/2 percentile
# TODO: This approach has several problems:
#     * Not necessarily correct question. In practice ppl use "Prediction Interval",
#       which guesses the interval where the next point is likely to land.
#       Estimation relatively complex, requires regressors
#     * Does not behave well if number of points is small
#     * If function strongly asymmetric, may introduce significant bias
def tolerance_interval(x, p):
    xNoNan = x[~np.isnan(x)]
    return np.percentile(xNoNan, [p/2, 1 - p/2])   # Cut off


# According to Bonferroni, p-value is a multiple of number of hypotheses that have been tested
# However, p-value may not exceed 1, so crop it to 1. It is probably not precise for large p-values,
#  but that is also irrelevant, because hypotheses with large p-values would be rejected anyway
def bonferroni_correction(pMat, nHypothesis):
    pMatCorr = pMat * nHypothesis
    pMatCorr[pMatCorr > 1] = 1
    return pMatCorr


def mannwhitneyu_nan_aware(data1, data2):
    data1nonan = data1[~np.isnan(data1)]
    data2nonan = data2[~np.isnan(data2)]

    if (len(data1nonan) == 0) or (len(data2nonan) == 0) or (len(set(data1nonan) | set(data2nonan)) < 2):
        print("Warning: MannWhitneyU test had zero samples",
              len(data1nonan),
              len(data2nonan),
              len(set(data1nonan)),
              len(set(data2nonan)),
              data1nonan[0],
              data2nonan[0],
              )

        logPval = 0
    else:
        logPval = -np.log10(mannwhitneyu(data1nonan, data2nonan, alternative="two-sided")[1])
    return logPval, [len(data1nonan), len(data2nonan)]


def wilcoxon_nan_aware(data1, data2):
    data1flat = data1.flatten()
    data2flat = data2.flatten()

    nanIdx = np.isnan(data1flat) | np.isnan(data2flat)
    data1nonan = data1flat[~nanIdx]
    data2nonan = data2flat[~nanIdx]

    if (len(data1nonan) == 0) or (len(data2nonan) == 0) or (len(set(data1nonan) | set(data2nonan)) < 2):
        print("Warning: Wilcoxon test had zero samples",
              len(data1nonan),
              len(data2nonan),
              len(set(data1nonan)),
              len(set(data2nonan)),
              data1nonan[0],
              data2nonan[0],
              )

        logPval = 0
    else:
        logPval = -np.log10(wilcoxon(data1nonan, data2nonan)[1])
    nNoNan = np.sum(~nanIdx)
    return logPval, [nNoNan, nNoNan]


# def classification_accuracy_weighted(a, b):
#     '''
#     Computes best accuracy of classifying between scalar variables using a separating line
#     Compensates for disbalances in samples - aims to recover separability of equal classes
#
#     :param a: 1D numpy array of floats
#     :param b: 1D numpy array of floats
#     :return: scalar measure of accuracy in percent
#     '''
#
#     aEff = a[~np.isnan(a)]
#     bEff = b[~np.isnan(b)]
#
#     Na = len(aEff)
#     Nb = len(bEff)
#     assert (Na > 0) and (Nb > 0)
#
#     x = np.hstack([aEff, bEff])[:, None]
#     y = [0] * Na + [1] * Nb
#     w = [Nb] * Na + [Na] * Nb
#     clf = svm.SVC()
#     clf.fit(x, y, sample_weight=w)
#     yHat = clf.predict(x)
#     return accuracy_score(y, yHat)


def classification_accuracy_weighted(xArr, yArr, alternative='greater'):
    '''
    Computes best accuracy of classifying between scalar variables using a separating line
    Compensates for disbalances in samples - aims to recover separability of equal classes

    :param xArr: 1D numpy array of floats
    :param yArr: 1D numpy array of floats
    :return: scalar measure of accuracy in percent
    '''

    if alternative == 'greater':
        xEff = xArr[~np.isnan(xArr)]
        yEff = yArr[~np.isnan(yArr)]

        nX = len(xEff)
        nY = len(yEff)
        assert (nX > 0) and (nY > 0)

        # xEffRange = set(xEff[xEff <= np.max(yEff)])
        # if len(xEffRange) == 0:
        #     return 0

        acc = [(np.sum(xArr >= t) / nX + np.sum(yEff < t) / nY) / 2 for t in xEff]
        return np.max(acc)
    if alternative == 'lesser':
        return classification_accuracy_weighted(yArr, xArr, alternative='greater')
    elif alternative == 'two-sided':
        acc1 = classification_accuracy_weighted(xArr, yArr, alternative='greater')
        acc2 = classification_accuracy_weighted(yArr, xArr, alternative='greater')
        return max(acc1, acc2)
