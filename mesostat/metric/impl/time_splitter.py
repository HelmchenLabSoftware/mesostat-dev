import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, list_assert_get_uniform_shape


def split2D(data2D, nHist):
    '''
    :param data2D:  Input array of shape [nTrial, nTime]
    :param nHist:   number of timesteps of history to consider
    :return:        x - past values; y - future values

    Convert dataset into past and future values. Future is always for 1 timestep, past can be as along as nHist
    NOTE: currently, past dimensions are sorted from oldest to newest, so data[t-1] = x[-1]
    '''
    nTrial, nTime = data2D.shape
    if nTime < nHist + 1:
        raise ValueError("Autoregression of history", nHist, "requires", nHist+1, "timesteps. Got", nTime)

    x = np.array([data2D[:, i:i + nHist] for i in range(nTime - nHist)])    # (rs) -> (srw)
    x = numpy_merge_dimensions(x.transpose((1, 0, 2)), 0, 2)                # (srw) -> (r*s, w)
    y = data2D[:, nHist:].flatten()                                         # (rs) -> (r*s)
    return x, y


def split2D_non_unifrom(dataLst, nHist):
    # sample windows over time (rs) -> (rsw)
    rez = []
    for data1D in dataLst:
        nTime = len(data1D)
        if nTime < nHist + 1:
            raise ValueError("Autoregression of history", nHist, "requires", nHist + 1, "timesteps. Got", nTime)

        rez += [np.array([data1D[i:i + nHist] for i in range(nTime - nHist)])]

    x = np.concatenate(rez, axis=0)                         # (rsw) -> (r*s, w)
    y = np.hstack([data1D[nHist:] for data1D in dataLst])   # (rs) -> (r*s)
    return x, y


def split3D(data, nHist):
    '''

    :param data:    Input array of shape 'rps'
    :param nHist:   number of timesteps of history to consider
    :return:        x - past values; y - future values

    Convert dataset into past and future values. Future is always for 1 timestep, past can be as along as nHist
    NOTE: currently, past dimensions are sorted from oldest to newest, so data[t-1] = x[-1]
    '''

    nTrial, nChannel, nTime = data.shape

    if nTime < nHist + 1:
        raise ValueError("Autoregression of history", nHist, "requires", nHist+1, "timesteps. Got", data.shape[2])

    # sample windows over time (rps) -> (srpw)
    x = np.array([data[:, :, i:i + nHist] for i in range(nTime - nHist)])

    # shape transform for x :: (srpw) -> (r*s, p*w)
    x = x.transpose((1, 0, 2, 3))        # (srpw) -> (rspw)
    x = numpy_merge_dimensions(x, 2, 4)  # p*w
    x = numpy_merge_dimensions(x, 0, 2)  # r*s

    # Convert to windows
    # shape transform for y :: (rps) -> (r*s, p)
    y = data.transpose((0,2,1))[:, nHist:]
    y = numpy_merge_dimensions(y, 0, 2)

    return x, y


def split3D_non_uniform(dataLst, nHist):
    # Test that the number of channels is uniform
    nChannel = list_assert_get_uniform_shape(dataLst, axis=1)

    # sample windows over time (rps) -> (rspw)
    rez = []
    for data2D in dataLst:
        nTime = data2D.shape[1]
        if nTime < nHist + 1:
            raise ValueError("Autoregression of history", nHist, "requires", nHist + 1, "timesteps. Got", nTime)

        rez += [np.array([data2D[:, i:i + nHist] for i in range(nTime - nHist)])]


    # shape transform for x :: (srpw) -> (r*s, p*w)
    # x = x.transpose((0, 2, 3, 1))
    x = np.concatenate(rez, axis=0)   # (rspw) -> (r*s, p, w)
    x = numpy_merge_dimensions(x, 1, 3)  # p*w

    # Convert to windows
    # shape transform for y :: (rps) -> (r*s, p)
    y = [data2D[:, nHist:].T for data2D in dataLst]   # (rps) -> (rsp)
    y = np.concatenate(y, axis=0)                     # (rsp) -> (r*s, p)

    return x, y