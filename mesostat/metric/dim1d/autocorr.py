import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.signals.filter import zscore
from mesostat.stat.stat import mu_std


# Calculates the 1-sided autocorrelation of a discrete 1D dataset
# Returned dataset has same length as input, first value normalized to 1.
def autocorr_1D(x: np.array):
    N = len(x)

    # FIXME Cheat - replace NAN's with random normal numbers with same mean and variance
    xEff = np.copy(x)
    nanIdx = np.isnan(x)
    xEff[nanIdx] = np.random.normal(*mu_std(x), np.sum(nanIdx))

    #return np.array([np.nanmean(x[iSh:] * x[:N-iSh]) for iSh in range(N)])
    # Note: As of time of writing np.correlate does not correctly handle nan division
    return np.correlate(xEff, xEff, 'full')[N - 1:] / N


def autocorr_3D(data: np.array, settings: dict):
    '''
    :param data:        3D data of shape "rps"
    :param settings:    Extra settings.
    :return:            Autocorrelation. Length same as data
    TODO: Currently autocorrelation is averaged over other provided dimensions. Check if there is a more rigorous way
    '''

    if data.shape[2] <= 1:
        raise ValueError("Autocorrelation requires more than 1 timestep")

    # Convert to canonical form
    dataFlat = numpy_merge_dimensions(data, 0, 2)
    dataThis = zscore(dataFlat)
    return np.nanmean(np.array([autocorr_1D(d) for d in dataThis]), axis=0)


# Crop lists to shortest one along 3rd dimension, return as array
def _trunc_data(dataLst: list):
    nSampleMin = np.min([data.shape[1] for data in dataLst])
    return np.array([data[:, :nSampleMin] for data in dataLst])


# Apply autocorrelation to 3D dataset with variable number of samples by cropping tail
# dataLst shape [nTrial x (nChannel, nSample)]
def autocorr_trunc_3D(dataLst: list, settings: dict):
    return autocorr_3D(_trunc_data(dataLst), settings)


# Calculates autocorrelation of unit time shift. Can handle nan's
def autocorr_d1_3D(data: np.array, settings: dict):
    if data.shape[2] <= 1:
        raise ValueError("Autocorrelation requires more than 1 timestep")

    dataZpre = zscore(data[:, :, :-1].flatten())
    dataZpost = zscore(data[:, :, 1:].flatten())
    return np.nanmean(dataZpre * dataZpost)


def autocorr_d1_3D_non_uniform(dataLst: list, settings: dict):
    return autocorr_d1_3D(_trunc_data(dataLst), settings)
