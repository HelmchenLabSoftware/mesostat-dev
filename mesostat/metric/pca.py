import numpy as np
from sklearn.decomposition import PCA

from mesostat.stat.stat import discrete_CDF
from mesostat.metric.impl.infotheory import entropy_discrete_1D


def _numpy_drop_nan_rows(data2D):
    assert data2D.ndim == 2
    return data2D[~np.isnan(data2D).any(axis=1)]


def _numpy_drop_nan_cols(data2D):
    assert data2D.ndim == 2
    return data2D[:, ~np.isnan(data2D).any(axis=0)]


# data2D of shape 'ps'
# Compute how many eigenvalues are necessary to have the total explained variance 'alpha',
#    if the largest eigenvalues are counted first
def rank_smooth2D(data2D, settings):
    eps = settings['epsilon'] if 'epsilon' in settings else 0.99

    pca = PCA()
    pca.fit_transform(data2D)
    pdf = sorted(pca.explained_variance_ratio_)[::-1]   # Sort explained variance in descending order
    cdf = discrete_CDF(pdf)[1:]                         # Convert to CDF

    return np.sum(cdf <= eps)


# data3D of shape 'rps'
def rank_smooth3D(data3D, settings):
    return rank_smooth2D(np.hstack(data3D), settings)


def erank2D(data2D, settings):
    '''
    :param data2D: input data of shape (channels, samples)
    :return: scalar - effective rank
    '''
    allowBadData = ('allowBadData' in settings) and settings['allowBadData']

    data2Deff = _numpy_drop_nan_cols(data2D)

    nChannels, nSamples = data2Deff.shape

    if nChannels <= 1:
        return 1
    elif nSamples < nChannels:
        warningText = 'Attempted to estimate correlation for shape with too few samples ' + str(data2Deff.shape)
        if allowBadData:
            print(warningText)
            return np.nan
        else:
            raise ValueError('Attempted to estimate correlation for shape with too few samples', data2Deff.shape)
    else:
        corr = np.corrcoef(data2Deff)
        eig = np.linalg.eigvals(corr)
        eigNorm = eig / np.sum(eig)

        return np.exp(entropy_discrete_1D(eigNorm))


# data3D of shape 'rps'
def erank3D(data, settings):
    return erank2D(np.hstack(data), settings)