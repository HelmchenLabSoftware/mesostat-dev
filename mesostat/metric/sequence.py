import numpy as np

from scipy.stats import ttest_ind_from_stats
from scipy.stats.mstats import gmean
from scipy.ndimage import gaussian_filter

from mesostat.utils.arrays import list_assert_get_uniform_shape, set_list_shapes
from mesostat.stat.stat import convert_pmf
from mesostat.stat.connectomics import offdiag_1D
from mesostat.stat.moments import discrete_mean_var


def _test_data_consistent_run(data, settings, func, lowtrialResult):
    nTrial, nChannel, nTime = data.shape

    if nChannel <= 1:
        raise ValueError("need at least 2 channels to evaluate orderability")

    if nTrial <= 1:
        print("Warning: Orderability is only sensible for multiple trials")
        return lowtrialResult

    return func(data, settings)


def _test_data_consistent_run_non_uniform(data2DLst, settings, func, lowtrialResult):
    nTrial = len(data2DLst)
    nChannel = list_assert_get_uniform_shape(data2DLst, 0)
    nTimeMin = np.min(set_list_shapes(data2DLst, 1))

    if nChannel <= 1:
        raise ValueError("need at least 2 channels to evaluate orderability")

    if nTrial <= 1:
        print("Warning: Orderability is only sensible for multiple trials")
        return lowtrialResult

    return func(data2DLst, settings)


# # Determine baseline by cell. If not specified by user, use smallest value that the cell exhibits
# def _get_baselines(dataOrd, settings):
#     if "baseline" in settings.keys():
#         return settings["baseline"]
#     else:
#         baselineNaive = np.min(dataOrd, axis=(0,2))
#         degenerateIdx = np.std(dataOrd, axis=(0,2)) < 1.0E-10
#         baselineNaive[degenerateIdx] -= 1  # If all values in the trace are exactly the same, set distribution as uniform
#         return baselineNaive
#
#
# def _get_baselines_non_uniform(dataOrdLst, settings):
#     if "baseline" in settings.keys():
#         return settings["baseline"]
#     else:
#         dataStacked = np.hstack(dataOrdLst)   # [nNodes, nTimes * nTrials]
#         baselineNaive = np.min(dataStacked, axis=1)
#         degenerateIdx = np.std(dataStacked, axis=1) < 1.0E-10
#         baselineNaive[degenerateIdx] -= 1  # If all values in the trace are exactly the same, set distribution as uniform
#         return baselineNaive

####################################
# Temporal Moments
####################################

# normalized sum of CCDF, given PMF
def _temporal_moments_1D(y, stdFilter=None):
    nSample = len(y)
    x = np.arange(nSample) / nSample

    if np.std(y) < 1.0E-10:
        # baseline = np.mean(y) - 1
        # p = convert_pmf(y, baseline)

        # If the data is essentially a constant, we will get mu=0.5 and std=0
        # This is suboptimal because having exactly the same mean for multiple trials would break binary orderability
        # So we introduce tiny noise to the mean to ensure mu is equally below and above 0.5
        return np.random.normal(0.5, 1.0E-7), 0
    else:
        # After much experimentation, this seems to be most stable for estimating mean and variance
        # From a noisy probability distribution
        if stdFilter is not None:
            yFiltered = gaussian_filter(y, stdFilter)
        else:
            yFiltered = y
        baseline = np.mean(yFiltered)
        p = convert_pmf(yFiltered, baseline)

    if np.any(np.isnan(p)):
        raise ValueError("Ups")

    mu, var = discrete_mean_var(x, p)

    # Symmetry breaking: If we get almost degenerate datasets, it is somehow still possible to get exactly the same means
    # To avoid this, add very weak random noise to the metric so that
    mu += np.random.normal(0, 1.0E-7)
    return mu, var


def temporal_moments_3D(data, settings, axis=(0, 1), noAvg=False):
    if data.shape[2] < 2:
        raise ValueError("Must have more than one timestep")

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    stdFilter = None if 'stdFilter' not in settings.keys() else settings['stdFilter']
    moments = [[_temporal_moments_1D(d, stdFilter=stdFilter) for d in dChannel] for dChannel in data]

    if noAvg:
        return np.array(moments)
    else:
        # Compute Mean of the data and Variance of the data (NOT variance of the mean!!!)
        # Mean of gaussian random variables is given by (mu, var) = (mean(mu_i), mean(var_i))
        return np.mean(moments, axis=axis)


def temporal_moments_3D_non_uniform(dataLst, settings, axis=(0, 1), noAvg=False):
    stdFilter = None if 'stdFilter' not in settings.keys() else settings['stdFilter']

    rezLst = []
    for data2D in dataLst:
        if data2D.shape[1] < 2:
            raise ValueError("Must have more than one timestep")
        rezLst += [[_temporal_moments_1D(d, stdFilter=stdFilter) for d in data2D]]

    if noAvg:
        # [nTrial x nChannel x 2]
        return np.array(rezLst)
    else:
        return np.mean(rezLst, axis=axis)


def temporal_mean_3D(data, settings, axis=(0,1), noAvg=False):
    return temporal_moments_3D(data, settings, axis=axis, noAvg=noAvg)[..., 0]


def temporal_mean_3D_non_uniform(data, settings, axis=(0,1), noAvg=False):
    return temporal_moments_3D_non_uniform(data, settings, axis=axis, noAvg=noAvg)[..., 0]

####################################
# Orderability
####################################

def _bivariate_binary_orderability_from_moments(mu):
    '''
    For each pair of channels, decide which one is earlier by comparing their time courses, converted to CDF
    :param mu: 1D Array of temporal means for every channel
    :return: 2D array of shape [nCell, nCell],
    '''

    biasRowExtrude = np.tile(mu, (len(mu), 1))
    rez = biasRowExtrude - biasRowExtrude.T

    # Check that there are no off-diagonal differences that evaluated exactly to zero.
    assert np.sum(offdiag_1D(rez) == 0) == 0

    return rez > 0


def _bivariate_student_orderability_from_moments(mu, var, nTrial, type="stat"):
    '''
    :param mu:       1D Array of temporal means for every channel
    :param var:      1D Array of temporal variances for every channel
    :param nTrial:   number of trials used to compute temporal moments. Necessary for T-test
    :param type:     Depending on output type, returns either T-Test statistic or p-value
    :return:         Scalar orderability index
    '''

    rezidx = 0 if type == "stat" else 1

    nNode = len(mu)
    std = np.sqrt(var)

    rez = np.full((nNode, nNode), np.nan)
    for i in range(nNode):
        for j in range(i+1, nNode):
            rez[i][j] = ttest_ind_from_stats(mu[i], std[i], nTrial, mu[j], std[j], nTrial, equal_var=False)[rezidx]
            rez[j][i] = rez[i][j]

    return rez


def bivariate_orderability_from_temporal_mean(temporalMeans2D, settings):
    '''
    :param temporalMeans2D:    temporal mean array of shape [nTrial x nChannel]
    :return:                   bivariate orderability of shape [nChannel, nChannel]

    Compute mean for each channel, then apply linear transformation and take absolute value
    '''

    directed = "directed" in settings.keys() and settings["directed"]

    # Frequency with which first cell later than second
    phat2D = np.nanmean([_bivariate_binary_orderability_from_moments(mu) for mu in temporalMeans2D], axis=0)

    ord = 2 * phat2D - 1
    if not directed:
        ord = np.abs(ord)

    ord[np.eye(len(ord), dtype=bool)] = np.nan
    return ord


def bivariate_binary_orderability_3D(data, settings):
    def _aux(data, settings):
        # baselines = _get_baselines(dataOrd, settings)
        temporalMeans2D = temporal_mean_3D(data, settings, noAvg=True)
        return bivariate_orderability_from_temporal_mean(temporalMeans2D, settings)

    return _test_data_consistent_run(data, settings, _aux, 0.5)


def bivariate_binary_orderability_3D_non_uniform(data2DLst, settings):
    def _aux(data2DLst, settings):
        # baselines = _get_baselines_non_uniform(data2DLst, settings)
        temporalMeans2D = temporal_mean_3D_non_uniform(data2DLst, settings, noAvg=True)
        return bivariate_orderability_from_temporal_mean(temporalMeans2D, settings)

    return _test_data_consistent_run_non_uniform(data2DLst, settings, _aux, 0.5)


def bivariate_student_orderability_3D(data, settings):
    def _aux(data, settings):
        # baselines = _get_baselines(dataOrd, settings)
        # [nTrial, nChannel, 2], where 2 stands for (mean, var)
        mu, var = temporal_moments_3D(data, settings, axis=0).T
        return _bivariate_student_orderability_from_moments(mu, var, len(data), type="stat")

    return _test_data_consistent_run(data, settings, _aux, 0)


def bivariate_student_orderability_3D_non_uniform(data2DLst, settings):
    def _aux(data2DLst, settings):
        # baselines = _get_baselines_non_uniform(data2DLst, settings)
        mu, var = temporal_moments_3D_non_uniform(data2DLst, settings, axis=0).T
        return _bivariate_student_orderability_from_moments(mu, var, len(data2DLst), type="stat")

    return _test_data_consistent_run_non_uniform(data2DLst, settings, _aux, 0.5)


def avg_bivariate_binary_orderability_3D(data, settings):
    return np.nanmean(offdiag_1D(bivariate_binary_orderability_3D(data, settings)))


def avg_bivariate_binary_orderability_3D_non_uniform(data2DLst, settings):
    return np.nanmean(offdiag_1D(bivariate_binary_orderability_3D_non_uniform(data2DLst, settings)))


def avg_bivariate_binary_orderability_from_temporal_mean(temporalMeans2D):
    return np.nanmean(offdiag_1D(bivariate_orderability_from_temporal_mean(temporalMeans2D, {})))


def avg_bivariate_student_orderability_3D(data, settings):
    def _aux(data, settings):
        mu, var = temporal_moments_3D(data, settings, axis=0).T
        pvals = _bivariate_student_orderability_from_moments(mu, var, len(data), type="pval")
        return gmean(offdiag_1D(pvals))

    return _test_data_consistent_run(data, settings, _aux, 1)


def avg_bivariate_student_orderability_3D_non_uniform(data2DLst, settings):
    def _aux(data2DLst, settings):
        # baselines = _get_baselines_non_uniform(data2DLst, settings)
        mu, var = temporal_moments_3D_non_uniform(data2DLst, settings, axis=0).T
        pvals = _bivariate_student_orderability_from_moments(mu, var, len(data2DLst), type="pval")
        return gmean(offdiag_1D(pvals))

    return _test_data_consistent_run_non_uniform(data2DLst, settings, _aux, 0.5)
