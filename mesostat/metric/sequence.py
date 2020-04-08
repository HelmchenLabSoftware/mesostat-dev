import numpy as np

from mesostat.utils.arrays import numpy_transpose_byorder
from mesostat.stat.stat import discrete_CDF, convert_pmf
from mesostat.stat.connectomics import offdiag_1D

# For each pair of channels, decide which one is earlier by comparing their time courses, converted to CDF
# Result is bool [nCell, nCell] array
def binary_cumul_ord_2D(data2D, baselineByCell):
    nNode, nTime = data2D.shape

    # Convert traces to probability distributions
    pvec = [convert_pmf(dataCell, baseline) for dataCell, baseline in zip(data2D, baselineByCell)]
    cdfVec = [discrete_CDF(p) for p in pvec]

    rez = np.zeros((nNode, nNode), dtype=bool)
    for i in range(nNode):
        for j in range(i+1, nNode):
            rez[i, j] = np.sum(cdfVec[i] - cdfVec[j]) > 0
            rez[j, i] = rez[i, j]

    return rez


# For each pair of channels, compute orderability metric
#   * Metric in [-1, 1]
#   * {0: order is undecided, 1: first cell later, -1: second cell later}
def cumul_ord_3D(data, settings):
    if "".join(sorted(settings['dim_order'])) != "prs":
        raise ValueError("Cumulative Orderability requires 3D data")

    # Transpose dataset into comfortable form
    dataOrd = numpy_transpose_byorder(data, settings['dim_order'], 'rps')

    # Determine baseline by cell. If not specified by user, use smallest value that the cell exhibits
    if "baseline" in settings.keys():
        baselineByCell = settings["baseline"]
    else:
        baselineByCell = np.min(dataOrd, axis=(0,2))

    ordByTrial = np.array([binary_cumul_ord_2D(dataTrial, baselineByCell) for dataTrial in dataOrd])
    phat2D = np.mean(ordByTrial, axis=0)  # Frequency with which first cell later than second
    return np.abs(2 * phat2D - 1)


# For each pair of channels, compute orderability metric
#   * Metric in [-1, 1]
#   * {0: order is undecided, 1: first cell later, -1: second cell later}
# DataShape [nTrial, nCell, nTime]. nCell is fixed, but nTime may vary between trials, hence data passed as list of 2D arrays
# Result is float [nCell, nCell] array
def cumul_ord_3D_non_uniform(data2DLst, settings):
    if "".join(sorted(settings['dim_order'])) != "ps":
        raise ValueError("Cumulative Orderability requires 3D data")
    assert np.all([np.prod(data.shape) != 0 for data in data2DLst]), "Should have non-zero data axis"

    if settings['dim_order'] != "ps":
        dataOrd = [numpy_transpose_byorder(data, settings['dim_order'], "ps") for data in data2DLst]
    else:
        dataOrd = data2DLst

    # Determine baseline by cell
    baselineByCell = np.array([np.min(dataTrial, axis=1) for dataTrial in dataOrd])
    baselineByCell = np.min(baselineByCell, axis=0)

    ordByTrial = np.array([binary_cumul_ord_2D(dataTrial, baselineByCell) for dataTrial in dataOrd])
    phat2D = np.mean(ordByTrial, axis=0)  # Frequency with which first cell later than second
    return np.abs(2 * phat2D - 1)


def avg_cumul_ord_3D(data, settings):
    return np.nanmean(offdiag_1D(cumul_ord_3D(data, settings)))


def avg_cumul_ord_3D_non_uniform(data2DLst, settings):
    return np.nanmean(offdiag_1D(cumul_ord_3D_non_uniform(data2DLst, settings)))
