import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
from mesostat.stat.machinelearning import drop_nan_rows

# Convert a 2D dataset to predicted values (current timesteps flattened) and predictors (past timesteps flattened)
# Afterwards, drop all rows where at least one value is NAN
# NOTE: currently, past dimensions are sorted from oldest to newest, so data[t-1] = x[-1]
def split2D(data2D, nHist):
    # TODO: Maybe this operation can be accelerated
    sweepTime = lambda d: np.array([d[iTime:iTime + nHist] for iTime in range(len(d) - nHist)])

    y = np.hstack([dataTrial[nHist:] for dataTrial in data2D])
    x = np.vstack([sweepTime(dataTrial) for dataTrial in data2D])

    # Truncate all datapoints that have at least one NAN in them
    return drop_nan_rows(x, y)


# Def
def split3D(data, dimOrder, nHist):
    test_have_dim("time_split", dimOrder, "s")

    dataCanon = numpy_transpose_byorder(data, dimOrder, 'srp', augment=True)
    nTime = dataCanon.shape[0]

    x = np.array([dataCanon[i:i + nHist] for i in range(nTime - nHist)])

    # shape transform for x :: (swrp) -> (s*r, p*w)
    x = x.transpose((0, 2, 3, 1))
    x = numpy_merge_dimensions(x, 2, 4)  # p*w
    x = numpy_merge_dimensions(x, 0, 2)  # s*r

    # shape transform for y :: (srp) -> (s*r, p)
    y = numpy_merge_dimensions(dataCanon[nHist:], 0, 2)

    return drop_nan_rows(x, y)


# # Construct stacked past measurements
# def AR_STACK_LAG(data, nHist):
#     nCh, nTr, nT = data.shape
#
#     x = np.zeros((nCh * nHist, nTr, nT - nHist))
#     for i in range(nHist):
#         x[nCh * i:nCh * (i + 1)] = data[:, :, i:i - nHist]
#
#     y = data[:, :, nHist:]
#
#     return x, y