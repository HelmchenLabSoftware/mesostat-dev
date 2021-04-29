import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AffinityPropagation, SpectralClustering, OPTICS

from mesostat.visualization.mpl_matrix import imshow
from mesostat.stat.connectomics import offdiag_1D


# Compute clustering given distance matrix and distance threshold
def cluster_dist_matrix(M, t, method='hierarchic'):
    if method == 'hierarchic':
        distTril = np.tril(M, 1)
        linkageMatrix = linkage(distTril, method='centroid', metric='euclidean', optimal_ordering=True)
        return fcluster(linkageMatrix, t, criterion='maxclust')  # - 1  # Original numbering starts at 1 for some reason
    #         linkageMatrix = linkage(distTril, method='centroid', metric='euclidean')
    #         rez = fcluster(linkageMatrix, t, criterion='distance')
    elif method == 'affinity':
        # clustering = AffinityPropagation(affinity='precomputed', preference=t).fit(M)  #damping=t
        clustering = AffinityPropagation(affinity='euclidean', preference=t).fit(M)  # damping=t
        rez = clustering.labels_
    elif method == 'spectral':
        clustering = SpectralClustering(affinity='precomputed', assign_labels="discretize", n_init=100).fit(M)
        rez = clustering.labels_
    elif method == 'optics':
        clustering = OPTICS(metric='precomputed', min_samples=t).fit(M)
        rez = clustering.labels_
    else:
        raise ValueError("Unknown method", method)

    # Original numbering may start at something other than 0 for some methods
    rez = np.array(rez, dtype=int)
    return rez - np.min(rez).astype(int)


def cluster_plot(fig, ax, M, clusters, channelLabels=None):
    idxs = np.argsort(clusters)
    MSort = M[idxs][:, idxs]

    idCluster, nCluster = np.unique(clusters, return_counts=True)
    nClustCum = np.cumsum(nCluster)

    imshow(fig, ax, MSort, 'clustering', limits=[-1,1], haveColorBar=True, cmap='jet')

    for nLine in nClustCum:
        ax.axvline(x=nLine - 0.5, linestyle='--', color='black', alpha=0.3)
        ax.axhline(y=nLine - 0.5, linestyle='--', color='black', alpha=0.3)

    if channelLabels is not None:
        cv = cluster_values(M, clusters)
        for iClust in sorted(set(clusters)):
            labelsThis = channelLabels[clusters==iClust]
            print(iClust, ':', cv['avgCorr'][iClust], cv['avgRelCorr'][iClust], ':', labelsThis)


def cluster_values(M, clusters):
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
