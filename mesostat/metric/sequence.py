import numpy as np

from scipy.stats import ttest_ind_from_stats
from scipy.stats.mstats import gmean
from scipy.ndimage import gaussian_filter

from mesostat.utils.arrays import numpy_transpose_byorder, test_have_dim, get_uniform_dim_shape, get_list_shapes
from mesostat.stat.stat import discrete_CDF, convert_pmf
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
    nChannel = get_uniform_dim_shape(data2DLst, 0)
    nTimeMin = np.min(get_list_shapes(data2DLst, 1))

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

    mu, std = discrete_mean_var(x, p)

    # Symmetry breaking: If we get almost degenerate datasets, it is somehow still possible to get exactly the same means
    # To avoid this, add very weak random noise to the metric so that
    mu += np.random.normal(0, 1.0E-7)
    return mu, std


def temporal_moments_3D(data, settings, axis=(0, 1)):
    if data.shape[2] < 2:
        raise ValueError("Must have more than one timestep")

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    stdFilter = None if 'stdFilter' not in settings.keys() else settings['stdFilter']
    moments = [[_temporal_moments_1D(d, stdFilter=stdFilter) for d in dChannel] for dChannel in data]

    # Compute Mean of the data and Variance of the data (NOT variance of the mean!!!)
    # Mean of gaussian random variables is given by (mu, var) = (mean(mu_i), mean(var_i))
    return np.mean(moments, axis=axis)


def temporal_moments_3D_non_uniform(dataLst, settings, axis=(0, 1)):
    stdFilter = None if 'stdFilter' not in settings.keys() else settings['stdFilter']

    rezLst = []
    for data2D in dataLst:
        if data2D.shape[1] < 2:
            raise ValueError("Must have more than one timestep")
        rezLst += [[_temporal_moments_1D(d, stdFilter=stdFilter) for d in data2D]]
    return np.mean(rezLst, axis=axis)


def temporal_mean_3D(data, settings):
    return temporal_moments_3D(data, settings)[0]


def temporal_mean_3D_non_uniform(data, settings):
    return temporal_moments_3D_non_uniform(data, settings)[0]


# For each pair of channels, decide which one is earlier by comparing their time courses, converted to CDF
# Result is bool [nCell, nCell] array
def bivariate_binary_orderability_2D(data2D, stdFilter=None):
    nNode, nTime = data2D.shape

    if nTime <= 2:
        print("WARNING: need multiple timesteps to evaluate orderability, returning NAN")
        return np.full((nNode, nNode), np.nan)

    moments = [_temporal_moments_1D(data, stdFilter=stdFilter) for data in data2D]
    mu, var = np.array(moments).T

    biasRowExtrude = np.tile(mu, (nNode, 1))
    rez = biasRowExtrude - biasRowExtrude.T

    # Check that there are no off-diagonal differences that evaluated exactly to zero.
    assert np.sum(offdiag_1D(rez) == 0) == 0

    return rez > 0


def _bivariate_student_orderability_from_moments(mu, var, nTrial, type="stat"):
    # Can return either T-Test statistic or p-value
    rezidx = 0 if type == "stat" else 1

    nNode = len(mu)
    std = np.sqrt(var)

    rez = np.full((nNode, nNode), np.nan)
    for i in range(nNode):
        for j in range(i+1, nNode):
            rez[i][j] = ttest_ind_from_stats(mu[i], std[i], nTrial, mu[j], std[j], nTrial, equal_var=False)[rezidx]
            rez[j][i] = rez[i][j]

    return rez


def bivariate_student_orderability_2D(data2D, stdFilter=None):
    moments = [_temporal_moments_1D(data, stdFilter=stdFilter) for data in data2D]
    mu, var = np.array(moments).T
    return _bivariate_student_orderability_from_moments(mu, var, 1, type="stat")


# Compute BO given a list of orderability matrices
# Basically compute mean for each channel, then apply linear transformation and take absolute value
def _bivariate_orderability_from_binary(ordByTrial):
    phat2D = np.nanmean(ordByTrial, axis=0)  # Frequency with which first cell later than second
    ord = np.abs(2 * phat2D - 1)
    ord[np.eye(len(ord), dtype=bool)] = np.nan
    return ord


def bivariate_binary_orderability_3D(data, settings):
    def _aux(data, settings):
        stdFilter = None if 'stdFilter' not in settings.keys() else settings['stdFilter']
        # baselines = _get_baselines(dataOrd, settings)
        ordByTrial = np.array([bivariate_binary_orderability_2D(dataTrial, stdFilter) for dataTrial in data])
        return _bivariate_orderability_from_binary(ordByTrial)

    return _test_data_consistent_run(data, settings, _aux, 0.5)


def bivariate_binary_orderability_3D_non_uniform(data2DLst, settings):
    def _aux(data2DLst, settings):
        stdFilter = None if 'stdFilter' not in settings.keys() else settings['stdFilter']
        # baselines = _get_baselines_non_uniform(data2DLst, settings)
        ordByTrial = np.array([bivariate_binary_orderability_2D(dataTrial, stdFilter) for dataTrial in data2DLst])
        return _bivariate_orderability_from_binary(ordByTrial)

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


# def _test_avg_bivar_bin_ord(ordByTrial, settings, nSample=2000):
#     metricFunc = lambda ord: np.nanmean(offdiag_1D(_bivariate_orderability_from_binary(ord)))
#
#     aboTrue = metricFunc(ordByTrial)
#     if "havePValue" not in settings or not settings["settings"]:
#         return aboTrue
#     else:
#         # Calculate p-value by means of permutation
#         ordByTrialFlat = ordByTrial.flatten()
#         aboTestLst = []
#         for iSample in range(nSample):
#             perm = np.random.permutation(len(ordByTrialFlat))
#             dataTest = ordByTrialFlat[perm].reshape(ordByTrial.shape)
#             aboTestLst += [metricFunc(dataTest)]
#
#         pval = np.max([np.mean(np.array(aboTestLst) > aboTrue), 1 / nSample])
#         return np.array([aboTrue, pval])


def avg_bivariate_binary_orderability_3D(data, settings):
    return np.nanmean(offdiag_1D(bivariate_binary_orderability_3D(data, settings)))


def avg_bivariate_binary_orderability_3D_non_uniform(data2DLst, settings):
    return np.nanmean(offdiag_1D(bivariate_binary_orderability_3D_non_uniform(data2DLst, settings)))


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
