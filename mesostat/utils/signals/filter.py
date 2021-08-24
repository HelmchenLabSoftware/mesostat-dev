import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA

from mesostat.utils.arrays import slice_sorted, numpy_shape_reduced_axes, numpy_move_dimension, numpy_merge_dimensions
from mesostat.stat.stat import gaussian


def zscore(x, axis=None):
    shapeNew = numpy_shape_reduced_axes(x.shape, axis)
    mu = np.nanmean(x, axis=axis).reshape(shapeNew)
    std = np.nanstd(x, axis=axis).reshape(shapeNew)
    return (x - mu) / std


def zscore_dim_ord(x, dimOrdSrc, dimOrdZ=None):
    if dimOrdZ is not None:
        axisZScore = tuple([i for i, e in enumerate(dimOrdSrc) if e in dimOrdZ])
        return zscore(x, axisZScore)
    else:
        return x


# ZScore list of arrays, computing mean and std from concatenated data
def zscore_list(lst):
    xFlat = np.hstack([data.flatten() for data in lst])
    mu = np.nanmean(xFlat)
    std = np.nanstd(xFlat)
    return [(data - mu)/std for data in lst]


# Remove first few principal components from data. Returns data in original space (not PCA space)
def drop_PCA(dataRP, nPCA=1):
    nChannel = dataRP.shape[1]
    pca = PCA(n_components=nChannel)
    nanIdxs = np.any(np.isnan(dataRP), axis=1)
    haveNan = np.any(nanIdxs)
    dataRPeff = dataRP if not haveNan else dataRP[~nanIdxs]

    if len(dataRPeff) <= nChannel:
        raise ValueError('Data has too few repetitions to estimate PCA', dataRP.shape, 'of those non-nan', dataRPeff.shape)

    y = pca.fit_transform(dataRPeff)
    M = pca.components_
    y[:, :nPCA] = 0  # Set the first nPCA components to zero in PCA space
    rez = y.dot(M)  # Map back from PCA space to original space

    if haveNan:
        rezNan = np.full(dataRP.shape, np.nan)
        rezNan[~nanIdxs] = rez
        return rezNan
    else:
        return rez


# Drop PCA computing it over all samples S and repetitions R. Returned shape same as input
def drop_PCA_3D(dataRSP, nPCA=1):
    dataRP = numpy_merge_dimensions(dataRSP, 0, 2)
    dataRPPCA = drop_PCA(dataRP, nPCA=nPCA)
    return dataRPPCA.reshape(dataRSP.shape)


# Compute discretized exponential decay convolution
# Works with multidimensional arrays, as long as shapes are the same
def approx_decay_conv(data, tau, dt):
    dataShape = data.shape
    nTimesTmp = dataShape[0] + 1  # Temporary data 1 longer because recursive formula depends on past
    tmpShape = (nTimesTmp,) + dataShape[1:]

    alpha = dt / tau
    beta = 1 - alpha

    rez = np.zeros(tmpShape)
    for i in range(1, nTimesTmp):
        rez[i] = data[i - 1] * alpha + rez[i - 1] * beta

    return rez[1:]  # Remove first element, because it is zero and meaningless. Get same shape as original data
