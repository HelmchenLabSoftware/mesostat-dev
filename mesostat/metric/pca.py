import numpy as np
from sklearn.decomposition import PCA

from mesostat.stat.stat import discrete_CDF
from mesostat.metric.impl.infotheory import entropy_discrete_1D


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


def erank2D(data2D):
    '''
    :param data2D: input data of shape (channels, samples)
    :return: scalar - effective rank
    '''

    if len(data2D) <= 1:
        return 1
    else:
        corr = np.corrcoef(data2D)
        eig = np.linalg.eigvals(corr)
        eigNorm = eig / np.sum(eig)

        return np.exp(entropy_discrete_1D(eigNorm))


# data3D of shape 'rps'
def erank3D(data, settings):
    return erank2D(np.hstack(data))