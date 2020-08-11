import numpy as np
from sklearn.decomposition import PCA

from mesostat.stat.stat import discrete_CDF


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