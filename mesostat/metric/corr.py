import numpy as np
import scipy.stats

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, get_list_shapes, get_uniform_dim_shape
from mesostat.stat.connectomics import offdiag_1D


# Test that along a given dimension all shapes are equal
def test_uniform_dimension(dataLst, dataDimOrder, dimEqual):
    if dimEqual in dataDimOrder:
        idxSample = dataDimOrder.index(dimEqual)
        shapeArr = np.array([d.shape for d in dataLst]).T
        assert np.all(shapeArr[idxSample] == shapeArr[idxSample][0]), "All trials are required to have the same number of channels"


# p-value of a single correlation between two scalar variables
# Null hypothesis: Both variables are standard normal variables
# Problem 1: When evaluating corr matrix, not clear how to Bonferroni-correct, because matrix entries are not independent
# Problem 2: Frequently the question if interest is comparing two scenarios with non-zero correlation, as opposed to comparing one scenario to 0 baseline
def corr_significance(c, nData):
    t = c * np.sqrt((nData - 2) / (1 - c**2))
    t[t == np.nan] = np.inf
    return scipy.stats.t(nData).pdf(t)


# Correlation. Requires leading dimension to be channels
# If y2D not specified, correlation computed between channels of x
# If y2D is specified, correlation computed for x-x and x-y in a composite matrix
def corr_2D(x2D, y2D=None, est='corr'):
    nChannel, nData = x2D.shape

    if nChannel <= 1:
        raise ValueError("Correlation requires at least 2 channels, got", nChannel)

    if nData <= 1:
        raise ValueError("Correlation requires at least 2 samples, got", nData)

    if est == 'corr':
        return np.corrcoef(x2D, y2D)
    elif est == 'spr':
        spr, p = scipy.stats.spearmanr(x2D, y2D, axis=1)

        # SPR has this "great" idea of only returning 1 number if exactly 2 channels are used
        if (nChannel == 2) and (y2D is None):
            coeff2mat = lambda d, c: np.array([[d, c],[c, d]])
            #return [coeff2mat(1, spr), coeff2mat(0, p)]
            return coeff2mat(1, spr)
        else:
            return spr
    else:
        raise ValueError('unexpected estimator type', est)


# If data has trials, concatenate trials into single timeline when computing correlation
def corr_3D(data, settings):
    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, 'rps', 'psr')
    dataFlat = numpy_merge_dimensions(dataCanon, 1, 3)

    est = settings['estimator'] if 'estimator' in settings.keys() else 'corr'
    return corr_2D(dataFlat, est=est)


# Compute average absolute value off-diagonal correlation (synchr. coeff)
def avg_corr_3D(data, settings):
    M = corr_3D(data, settings)
    return np.nanmean(np.abs(offdiag_1D(M)))


def corr_3D_non_uniform(dataLst, settings):
    est = settings['estimator'] if 'estimator' in settings.keys() else 'corr'
    return corr_2D(np.hstack(dataLst), est=est)


def avg_corr_3D_non_uniform(dataLst, settings):
    M = corr_3D_non_uniform(dataLst, settings)
    return np.nanmean(np.abs(offdiag_1D(M)))



# FIXME: Correct all TE-based procedures, to compute cross-correlation as a window sweep externally
# FIXME: Adapt all TE-based procedures to use 1 lag at a time, or redefine extra procedure to use multiple lags
def cross_corr_3D(data, settings):
    '''
    Compute cross-correlation of multivariate dataset for a fixed lag

    :param data: 2D or 3D matrix
    :param settings: A dictionary. 'min_lag_sources' and 'max_lag_sources' determine lag range.
    :param est: Estimator name. Can be 'corr' or 'spr' for cross-correlation or spearmann-rank
    :return: A matrix [nLag x nSource x nTarget]
    '''

    # Transpose dataset into comfortable form
    dataOrd = numpy_transpose_byorder(data, 'rps', 'psr')

    # Extract parameters
    # Extract parameters
    lag = settings['lag']
    nNode, nTime = dataOrd.shape[:2]

    # Check that number of timesteps is sufficient to estimate lagMax
    if nTime <= lag:
        raise ValueError('lag', lag, 'cannot be estimated for number of timesteps', nTime)

    xx = numpy_merge_dimensions(dataOrd[:, :nTime - lag], 1, 3)
    yy = numpy_merge_dimensions(dataOrd[:, lag:], 1, 3)

    # Only interested in x-y correlations, crop x-x and y-y
    est = settings['estimator'] if 'estimator' in settings.keys() else 'corr'
    return corr_2D(xx, yy, est=est)[nNode:, :nNode]


def cross_corr_non_uniform_3D(dataLst, settings):
    '''
    Compute cross-correlation of multivariate dataset for a fixed lag

    :param dataLst: a list of 2D matrices. Effective shape "rps" or "rsp"
    :param settings: A dictionary. 'min_lag_sources' and 'max_lag_sources' determine lag range.
    :param est: Estimator name. Can be 'corr' or 'spr' for cross-correlation or spearmann-rank
    :return: A matrix [nLag x nSource x nTarget]
    '''

    # Extract parameters
    lag = settings['lag']

    # Test that all trials have the same number of
    # Test that all trials have sufficient timesteps for lag estimation
    nNode = get_uniform_dim_shape(dataLst, axis=0)
    nTimeMin = np.min(get_list_shapes(dataLst, axis=1))

    if nTimeMin <= lag:
        raise ValueError('lag', lag, 'cannot be estimated for number of timesteps', nTimeMin)

    xx = np.hstack([data[:, lag:] for data in dataLst])
    yy = np.hstack([data[:, :-lag] for data in dataLst])

    # Only interested in x-y correlations, crop x-x and y-y
    est = settings['estimator'] if 'estimator' in settings.keys() else 'corr'
    return corr_2D(xx, yy, est=est)[nNode:, :nNode]


# Correlation that works if some values in the dataset are NANs
def corr_nan(x2D):
    pass
    # z2D = zscore(x2D, axis=1)
    # nChannel, nData = x2D.shape
    # rez = np.ones((nChannel, nChannel))
    # for i in range(nChannel):
    #     for j in range(i+1, nChannel):
    #         rez[i][j] = np.nanmean(z2D[i] * z2D[j])
    #         rez[j][i] = rez[i][j]
    # return rez