import numpy as np
from scipy import linalg


def _residual_linear_fit(x, cov):
    coeffX = linalg.lstsq(cov.T, x)[0]
    return x - coeffX.dot(cov)


def r2(trg, srcLst):
    return 1 - np.var(_residual_linear_fit(trg, np.array(srcLst))) / np.var(trg)


def pr2_quadratic_triplet_decomp_1D(src1, src2, trg):
    '''
    :param src1:  First source channel, 1D array of floats
    :param src2:  Second source channel, 1D array of floats
    :param trg:   Target channel, 1D array of floats
    :return:      4 relative explained variances: unique source 1, unique source 2, redundant, synergistic

    Partial R^2-based decomposition of triplet correlations onto unique, redundant and synergistic parts
    Unique and redundant terms are purely linear, synergistic term is quadratic.
    '''

    # Compute first two centered moments
    c1 = src1 - np.mean(src1)
    c2 = src2 - np.mean(src2)
    cTrg = trg - np.mean(trg)
    c12 = c1 * c2

    # # Fit, compute ExpVariduals and related variances
    rev1 = r2(cTrg, [c1])
    rev2 = r2(cTrg, [c2])

    revFull = r2(cTrg, [c1, c2, c12])
    revM1   = r2(cTrg, [c2, c12])
    revM2   = r2(cTrg, [c1, c12])
    revM12  = r2(cTrg, [c1, c2])

    # Compute individual effects
    unq1 = revFull - revM1
    unq2 = revFull - revM2
    syn12 = revFull - revM12
    # red12 = revFull - unq1 - unq2 - syn12
    red12 = rev1 + rev2 - revM12

    return np.clip([unq1, unq2, red12, syn12], 0, None)  # Clip negative entries as they are meaningless
