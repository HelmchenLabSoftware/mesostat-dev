import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions


def nansum(data, settings):
    return np.nansum(data)


def nanmean(data, settings):
    return np.nanmean(data)


def nanstd(data, settings):
    return np.nanstd(data)


def varmean(data, settings):
    data2D = numpy_merge_dimensions(data, 1, 3).T  # (p*s, r)
    nDimEff = len(data2D)
    return np.sum(np.cov(data2D)) / nDimEff**2
