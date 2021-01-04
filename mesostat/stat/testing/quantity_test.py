import numpy as np

from scipy.stats import mannwhitneyu, wilcoxon, binom
rstest_twosided = lambda x, y : mannwhitneyu(x, y, alternative='two-sided')
from mesostat.stat.permtests import difference_test


def test_quantity(dataA, dataB, pval, proxyFunc=None, test='mannwhitneyu', nResample=1000):
    '''
    :param dataA:  2D numpy array [nObject, nSampleA] - samples of multiple objects under condition A
    :param dataB:  2D numpy array [nObject, nSampleB] - samples of multiple objects under condition B
    :param pval:   p-value of the resulting test
    :param proxyFunc:  if None, ranksum test is performed for each object. Otherwise proxyFunc(A - B) permutation test is performed for each object
    :param test:       type of the test (mannwhitneyu/wilcoxon)
    :param nResample:  Number of resample points for permutation test
    :return:
                pvalue for each object (not corrected for multiple comparisons);
                Number of objects that are significant given this test;
                Negative logarithm of the p-value of the population test for difference between the two conditions

    Counts in how many objects the value under condition A is significantly different than under condition B. Then,
    performs binomial test to identify whether the number of objects that turn out significant is greater than chance.

    '''

    assert len(dataA) == len(dataB)
    nObject = len(dataA)
    pValByTest = np.zeros(nObject)

    # We are testing the values directly via wilcoxon or mann-whitney u test
    if proxyFunc is None:
        test_func = wilcoxon if test == 'wilcoxon' else rstest_twosided
        for iObject in range(nObject):
            pValByTest[iObject] = test_func(dataA[iObject], dataB[iObject])[1]

    # We are testing a function of the values, for which true distribution is unknown
    # Instead use permutation testing
    else:
        for iObject in range(nObject):
            pless, pmore = difference_test(proxyFunc, dataA[iObject], dataB[iObject], nResample,
                                           sampleFunction="permutation")
            pValByTest[iObject] = np.min([pless, pmore])

    nObjectSignificant = np.sum(pValByTest < pval)
    negLogPValPop = binom_ccdf(nObject, nObjectSignificant, pval)

    return pValByTest, nObjectSignificant, negLogPValPop


def binom_ccdf(nObject, nObjectSignificant, pval):
    # Compute probability of seeing at least nObjectSignificant positive outcomes from nObject by chance
    # Given that all tests are probability pval of being true by chance
    binomPMF = binom.pmf(np.arange(0, nObject), nObject, pval)
    pValPop = np.sum(binomPMF[nObjectSignificant:])
    return -np.log10(pValPop)

