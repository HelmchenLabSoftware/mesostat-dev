
import numpy as np
import matplotlib.pyplot as plt
import numpy.random

from scipy.cluster.hierarchy import linkage, fcluster
from mesostat.visualization.mpl_matrix import imshow
from mesostat.stat.clustering import cluster_plot, cluster_dist_matrix_min
from sklearn.cluster import AffinityPropagation, SpectralClustering, OPTICS, AgglomerativeClustering


def hierarchic_clustering(M, t, method='single', criterion='maxclust'):
    distTril = np.tril(M, 1)
    linkageMatrix = linkage(distTril, method=method, optimal_ordering=True)
    return fcluster(linkageMatrix, t, criterion=criterion)

##############################
# Generate random points
##############################
nPointPerCluster = 20
x1 = [0, 1]
x2 = [1, 0]
x3 = [-0.5, -0.5]
dataPoints = np.array(nPointPerCluster*[x1, x2, x3])
np.random.shuffle(dataPoints)

##############################
# Generate distance matrix
##############################
# Note: In general, if one has access to coordinates, clusters can be created directly from coordinates
# However, in some cases only the distance matrix is available, but not the coordinates
# This is the case we are trying to simulate

nPoint = len(dataPoints)
distMat = np.zeros((nPoint, nPoint))
for i in range(nPoint):
    for j in range(nPoint):
        distMat[i][j] = np.linalg.norm(dataPoints[i] - dataPoints[j])

distMat += np.random.normal(0, 0.5, distMat.shape)
# distMat = -distMat
# distMat = 3 - distMat
distMat = np.clip(distMat, 0, None)



##############################
# Construct clustering
##############################
methodsDict = {
    'Hierarchic' : lambda M: hierarchic_clustering(M, 5.0, method='complete', criterion='maxclust'),
    'Affinity'   : lambda M: AffinityPropagation(affinity='precomputed', damping=0.5).fit(M).labels_,
    'Spectral'   : lambda M: SpectralClustering(affinity='precomputed', gamma=10).fit(M).labels_,
    # 'OPTICS'     : lambda M: OPTICS(metric='precomputed', min_samples=10).fit(M).labels_,
    'OPTICS'     : lambda M: cluster_dist_matrix_min(M, 0.05, method='OPTICS'),
    'Agglo'      : lambda M: AgglomerativeClustering(affinity='precomputed', n_clusters=4, linkage='single').fit(M).labels_
}


nCols = len(methodsDict) + 1
fig, ax = plt.subplots(ncols=nCols, figsize=(4*nCols, 4))
imshow(fig, ax[0], distMat, title='Raw', haveColorBar=True, cmap=None)

for i, (methodName, methodFunc) in enumerate(methodsDict.items()):
    cluster_plot(fig, ax[i+1], distMat, methodFunc(distMat), cmap=None, limits=None)
    ax[i+1].set_title(methodName)

plt.show()