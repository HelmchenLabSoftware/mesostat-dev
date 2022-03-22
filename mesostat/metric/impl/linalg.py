import numpy as np
from scipy import linalg


def residual_linear_fit(x, cov):
    '''
    :param x:     1D vector of i.i.d data samples [nSample]
    :param cov:   2D vector of covariates [nCov, nSample]
    :return:      residual of X after all covariates are fitted to it and subtracted from it
    '''

    coeffX = linalg.lstsq(cov.T, x)[0]
    return x - coeffX.dot(cov)
