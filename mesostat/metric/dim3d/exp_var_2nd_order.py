import numpy as np
from scipy import linalg


def _residual_linear_fit(x, cov):
    coeffX = linalg.lstsq(cov.T, x)[0]
    return x - coeffX.dot(cov)


def _relative_explained_variance(trg, srcLst):
    return 1 - np.var(_residual_linear_fit(trg, np.array(srcLst))) / np.var(trg)


def quadratic_triplet_decomp_1D(src1, src2, trg):
    '''
    :param src1:  First source channel, 1D array of floats
    :param src2:  Second source channel, 1D array of floats
    :param trg:   Target channel, 1D array of floats
    :return:      4 relative explained variances: unique source 1, unique source 2, redundant, synergistic
    '''

    # Compute first two centered moments
    c1 = src1 - np.mean(src1)
    c2 = src2 - np.mean(src2)
    cTrg = trg - np.mean(trg)
    c12 = c1 * c2

    # Fit, compute ExpVariduals and related variances
    rev1 = _relative_explained_variance(cTrg, [c1])
    rev2 = _relative_explained_variance(cTrg, [c2])
    rev12 = _relative_explained_variance(cTrg, [c1, c2])
    rev12sq = _relative_explained_variance(cTrg, [c1, c2, c12])

    # Compute individual effects
    red12 = rev1 + rev2 - rev12
    unq1 = rev1 - red12
    unq2 = rev2 - red12
    syn12 = rev12sq - rev12

    return [unq1, unq2, red12, syn12]
