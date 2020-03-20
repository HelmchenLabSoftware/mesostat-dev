import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
from mesostat.utils.signals import zscore
from mesostat.stat.stat import mu_std

# Calculates the 1-sided autocorrelation of a discrete 1D dataset
# Returned dataset has same length as input, first value normalized to 1.
def autocorr_1D(x):
    N = len(x)

    # FIXME Cheat - replace NAN's with random normal numbers with same mean and variance
    xEff = np.copy(x)
    nanIdx = np.isnan(x)
    xEff[nanIdx] = np.random.normal(*mu_std(x), np.sum(nanIdx))

    #return np.array([np.nanmean(x[iSh:] * x[:N-iSh]) for iSh in range(N)])
    # Note: As of time of writing np.correlate does not correctly handle nan division
    return np.correlate(xEff, xEff, 'full')[N - 1:] / N


# Calculates autocorrelation. Any dimensions
# TODO: Currently autocorrelation is averaged over other provided dimensions. Check if there is a more rigorous way
def autocorr_3D(data, settings):
    test_have_dim("autocorr_3D", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    dataThis = zscore(dataFlat)
    return np.nanmean(np.array([autocorr_1D(d) for d in dataThis]), axis=0)

# List of arrays, shape [nTrial, nSample]
def autocorr_trunc_1D(dataLst):
    nSampleMin = np.min([len(data) for data in dataLst])
    return np.nanmean([data[:nSampleMin] for data in dataLst], axis=0)

# Calculates autocorrelation of unit time shift. Can handle nan's
def autocorr_d1_3D(data, settings):
    test_have_dim("autocorr_d1_3D", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataZ = zscore(dataCanon)

    dataZpre  = dataZ[:-1].flatten()
    dataZpost = dataZ[1:].flatten()
    return np.nanmean(dataZpre * dataZpost)
