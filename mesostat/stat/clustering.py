import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AffinityPropagation, SpectralClustering, OPTICS, AgglomerativeClustering

from mesostat.visualization.mpl_matrix import imshow
from mesostat.utils.matrix import offdiag_1D


def cluster_dist_matrix_min(M, t, method='Agglomerative'):
    '''
    :param M:      Distance matrix
    :param t:      Parameter for the clustering algorithm
    :param method: Clustering algorithm
    :return:       Cluster labels for each point

    Below algorithms minimize distance between points within the cluster
    '''
    if method == 'Agglomerative':
        rez = AgglomerativeClustering(affinity='precomputed', n_clusters=t, linkage='single').fit(M).labels_
    elif method == 'OPTICS':
        rez = OPTICS(metric='precomputed', xi=t).fit(M).labels_
    else:
        raise ValueError("Unknown method", method)

    # Original numbering may start at something other than 0 for some methods
    rez = np.array(rez, dtype=int)
    return rez - np.min(rez).astype(int)


def cluster_dist_matrix_max(M, t, method='Affinity'):
    '''
    :param M:      Distance matrix
    :param t:      Parameter for the clustering algorithm
    :param method: Clustering algorithm
    :return:       Cluster labels for each point

    Below algorithms maximize distance between points within the cluster
    '''
    if method == 'Affinity':
        rez = AffinityPropagation(affinity='precomputed', damping=t).fit(M).labels_
    else:
        raise ValueError("Unknown method", method)

    # Original numbering may start at something other than 0 for some methods
    rez = np.array(rez, dtype=int)
    return rez - np.min(rez).astype(int)


def cluster_plot(fig, ax, M, clusters, channelLabels=None, limits=None, cmap='jet', haveColorBar=True):
    '''
    :param fig:            Matplotlib figure
    :param ax:             Matplotlib axis
    :param M:              Distance matrix
    :param clusters:       Cluster labels
    :param channelLabels:  Channel labels
    :param limits:         Limits for the distance matrix plot
    :param cmap:           Color map
    :return:

    Plot distance matrix heatmap sorted by cluster. Plot separating lines for clusters
    '''

    idxs = np.argsort(clusters)
    MSort = M[idxs][:, idxs]

    idCluster, nCluster = np.unique(clusters, return_counts=True)
    nClustCum = np.cumsum(nCluster)

    imshow(fig, ax, MSort, 'clustering', limits=limits, haveColorBar=haveColorBar, cmap=cmap)

    for nLine in nClustCum:
        ax.axvline(x=nLine - 0.5, linestyle='--', color='black', alpha=0.3)
        ax.axhline(y=nLine - 0.5, linestyle='--', color='black', alpha=0.3)

    if channelLabels is not None:
        cv = cluster_values(M, clusters)
        for iClust in sorted(set(clusters)):
            labelsThis = channelLabels[clusters==iClust]
            print(iClust, ':', cv['avgCorr'][iClust], cv['avgRelCorr'][iClust], ':', labelsThis)


def cluster_values(M, clusters):
    '''
    :param M:          Distance matrix
    :param clusters:   Cluster labels
    :return:           Dictionary of metrics of cluster fidelity. Each metric computed for each cluster

    avgCorr - average in-cluster distance
    avgRelCorr - average in-cluster distance minus out-cluster distance
    '''

    rez = {'avgCorr' : {}, 'avgRelCorr' : {}}

    for iClust in sorted(set(clusters)):
        idxClust = clusters == iClust

        clustBlock = M[idxClust][:, idxClust]
        clustNonclustBlock = M[~idxClust][:, idxClust]

        muClust = np.mean(offdiag_1D(clustBlock))
        muNon = np.mean(clustNonclustBlock)
        rez['avgCorr'][iClust] = muClust
        rez['avgRelCorr'][iClust] = muClust - muNon
    return rez
