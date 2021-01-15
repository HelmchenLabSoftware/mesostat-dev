import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(ax, data2D, labels=None, legend=True):
    pca = PCA(n_components=2)
    rez = pca.fit_transform(data2D)

    if labels is None:
        ax.plot(*rez.T, '.')
    else:
        for label in set(labels):
            labelIdxs = labels == label
            ax.plot(*rez[labelIdxs].T, '.', label=label)

    if legend:
        ax.legend()
