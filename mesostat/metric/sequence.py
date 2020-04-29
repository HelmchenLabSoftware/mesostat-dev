import numpy as np

from scipy.stats import ttest_ind_from_stats
from scipy.stats.mstats import gmean
from scipy.ndimage import gaussian_filter

from mesostat.utils.arrays import numpy_transpose_byorder, test_have_dim
from mesostat.stat.stat import discrete_CDF, convert_pmf
from mesostat.stat.connectomics import offdiag_1D
from mesostat.stat.moments import discrete_mean_var


# Convert 3D matrix to standard form
def _preprocess_3D(data, settings):
    if "".join(sorted(settings['dim_order'])) != "prs":
        raise ValueError("Cumulative Orderability requires 3D data")

    # Transpose dataset into comfortable form
    dataOrd = numpy_transpose_byorder(data, settings['dim_order'], 'rps')
    if len(dataOrd) <= 1:
        print("Warning: Orderability requires multiple trials")

    return dataOrd


def _preprocess_3D_non_uniform(data2DLst, settings):
    if "".join(sorted(settings['dim_order'])) != "ps":
        raise ValueError("Cumulative Orderability requires 3D data")
    assert np.all([np.prod(data.shape) != 0 for data in data2DLst]), "Should have non-zero data axis"

    if len(data2DLst) <= 1:
        print("Warning: Orderability requires multiple trials")

    if settings['dim_order'] != "ps":
        return [numpy_transpose_byorder(data, settings['dim_order'], "ps") for data in data2DLst]
    else:
        return data2DLst


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
def _temporal_moments_1D(y, stdFilter=5):
    nSample = len(y)
    x = np.arange(nSample) / nSample

    if np.std(y) < 1.0E-10:
        baseline = np.mean(y) - 1
        p = convert_pmf(y, baseline)
    else:
        # After much experimentation, this seems to be most stable for estimating mean and variance
        # From a noisy probability distribution
        yFiltered = gaussian_filter(y, stdFilter)
        baseline = np.mean(yFiltered)
        p = convert_pmf(yFiltered, baseline)

    if np.any(np.isnan(p)):
        raise ValueError("Ups")

    return discrete_mean_var(x, p)


def temporal_moments_3D(data, settings):
    test_have_dim("bivariate_timing_rank_3D", settings['dim_order'], "s")

    # Transpose dataset into comfortable form
    dataOrd = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)

    # baselines = _get_baselines(dataOrd, settings)

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    moments = [[_temporal_moments_1D(d) for d in dChannel] for dChannel in dataOrd]

    # Compute Mean of the data and Variance of the data (NOT variance of the mean!!!)
    # Mean of gaussian random variables is given by (mu, var) = (mean(mu_i), mean(var_i))
    return np.mean(moments, axis=(0, 1))[0]  # FIXME


def temporal_moments_3D_non_uniform(dataLst, settings):
    rezLst = [temporal_moments_3D(data, settings) for data in dataLst]
    return np.mean(rezLst, axis=0)


# For each pair of channels, decide which one is earlier by comparing their time courses, converted to CDF
# Result is bool [nCell, nCell] array
def bivariate_binary_orderability_2D(data2D):
    nNode, nTime = data2D.shape
    moments = [_temporal_moments_1D(data) for data in data2D]
    mu, var = np.array(moments).T

    biasRowExtrude = np.tile(mu, (nNode, 1))
    rez = biasRowExtrude - biasRowExtrude.T
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


def bivariate_student_orderability_2D(data2D):
    moments = [_temporal_moments_1D(data) for data in data2D]
    mu, var = np.array(moments).T
    return _bivariate_student_orderability_from_moments(mu, var, 1, type="stat")


# Compute BO given a list of orderability matrices
# Basically compute mean for each channel, then apply linear transformation and take absolute value
def _bivariate_orderability_from_binary(ordByTrial):
    phat2D = np.mean(ordByTrial, axis=0)  # Frequency with which first cell later than second
    ord = np.abs(2 * phat2D - 1)
    ord[np.eye(len(ord), dtype=bool)] = np.nan
    return ord


def bivariate_binary_orderability_3D(data, settings):
    dataOrd = _preprocess_3D(data, settings)
    # baselines = _get_baselines(dataOrd, settings)
    if len(dataOrd) <= 1:
        return 0.5

    ordByTrial = np.array([bivariate_binary_orderability_2D(dataTrial) for dataTrial in dataOrd])
    return _bivariate_orderability_from_binary(ordByTrial)


def bivariate_binary_orderability_3D_non_uniform(data2DLst, settings):
    dataOrdLst = _preprocess_3D_non_uniform(data2DLst, settings)
    # baselines = _get_baselines_non_uniform(data2DLst, settings)
    if len(dataOrdLst) <= 1:
        return 0.5

    ordByTrial = np.array([bivariate_binary_orderability_2D(dataTrial) for dataTrial in dataOrdLst])
    return _bivariate_orderability_from_binary(ordByTrial)


def bivariate_student_orderability_3D(data, settings):
    dataOrd = _preprocess_3D(data, settings)
    # baselines = _get_baselines(dataOrd, settings)

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    moments = [[_temporal_moments_1D(d) for d in dChannel] for dChannel in dataOrd]

    mu, var = np.mean(moments, axis=0).T
    return _bivariate_student_orderability_from_moments(mu, var, len(dataOrd), type="stat")


def bivariate_student_orderability_3D_non_uniform(data2DLst, settings):
    dataOrdLst = _preprocess_3D_non_uniform(data2DLst, settings)
    # baselines = _get_baselines_non_uniform(data2DLst, settings)

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    moments = [[_temporal_moments_1D(d) for d in dChannel] for dChannel in dataOrdLst]

    mu, var = np.mean(moments, axis=0).T
    return _bivariate_student_orderability_from_moments(mu, var, len(dataOrdLst), type="stat")


def _test_avg_bivar_bin_ord(ordByTrial, settings, nSample=2000):
    metricFunc = lambda ord: np.nanmean(offdiag_1D(_bivariate_orderability_from_binary(ord)))

    aboTrue = metricFunc(ordByTrial)
    if "havePValue" not in settings or not settings["settings"]:
        return aboTrue
    else:
        # Calculate p-value by means of permutation
        ordByTrialFlat = ordByTrial.flatten()
        aboTestLst = []
        for iSample in range(nSample):
            perm = np.random.permutation(len(ordByTrialFlat))
            dataTest = ordByTrialFlat[perm].reshape(ordByTrial.shape)
            aboTestLst += [metricFunc(dataTest)]

        pval = np.max([np.mean(np.array(aboTestLst) > aboTrue), 1 / nSample])
        return np.array([aboTrue, pval])


def avg_bivariate_binary_orderability_3D(data, settings):
    dataOrd = _preprocess_3D(data, settings)
    # baselines = _get_baselines(dataOrd, settings)
    ordByTrial = np.array([bivariate_binary_orderability_2D(dataTrial) for dataTrial in dataOrd])
    return _test_avg_bivar_bin_ord(ordByTrial, settings)


def avg_bivariate_binary_orderability_3D_non_uniform(data2DLst, settings):
    dataOrdLst = _preprocess_3D_non_uniform(data2DLst, settings)
    # baselines = _get_baselines_non_uniform(data2DLst, settings)
    ordByTrial = np.array([bivariate_binary_orderability_2D(dataTrial) for dataTrial in dataOrdLst])
    return _test_avg_bivar_bin_ord(ordByTrial, settings)


def avg_bivariate_student_orderability_3D(data, settings):
    dataOrd = _preprocess_3D(data, settings)
    # baselines = _get_baselines(dataOrd, settings)

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    moments = [[_temporal_moments_1D(d) for d in dChannel] for dChannel in dataOrd]

    mu, var = np.mean(moments, axis=0).T
    pvals = _bivariate_student_orderability_from_moments(mu, var, len(dataOrd), type="pval")
    return gmean(offdiag_1D(pvals))


def avg_bivariate_student_orderability_3D_non_uniform(data2DLst, settings):
    dataOrdLst = _preprocess_3D_non_uniform(data2DLst, settings)
    # baselines = _get_baselines_non_uniform(data2DLst, settings)

    # [nTrial, nChannel, 2], where 2 stands for (mean, var)
    moments = [[_temporal_moments_1D(d) for d in dChannel] for dChannel in dataOrdLst]

    mu, var = np.mean(moments, axis=0).T
    pvals = _bivariate_student_orderability_from_moments(mu, var, len(dataOrdLst), type="pval")
    return gmean(offdiag_1D(pvals))