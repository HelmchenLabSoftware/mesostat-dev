import numpy as np

from mesostat.metric.impl.basis_projector import BasisProjector
from mesostat.utils.signals import resample


# Evaluate first few projections to Legendre basis
def basis_projection_1D(data, settings):
    nSample = len(data)
    order = settings["basisOrder"] if "basisOrder" in settings else 5
    bp = BasisProjector(nSample, order=order)
    return bp.project(data)


def stretch_basis_projection(data, settings):
    if data.shape[2] < 2:
        raise ValueError("Expected more than 1 timestep")

    # Average all axis except samples
    dataAvg = np.nanmean(data, axis=(0, 1))

    return basis_projection_1D(dataAvg, settings)


def stretch_basis_projection_non_uniform(dataLst, settings):
    rez = []
    for data2D in dataLst:
        if data2D.shape[1] < 2:
            raise ValueError("Expected more than 1 timestep")

        data1D = np.nanmean(data2D, axis=0)
        rez += [basis_projection_1D(data1D, settings)]

    return np.mean(rez, axis=0)


# Resample data to fixed length
def resample_non_uniform(data2DLst, settings):
    nTimeTrg = settings["nResamplePoint"]
    xTrg = np.linspace(0, 1, nTimeTrg)
    paramResample = {"method" : "downsample", "kind" : "kernel"}

    rezLst = []
    for data2D in data2DLst:
        if np.any(np.isnan(data2D)):
            raise ValueError("Can't resample data that has NAN in it")

        # Average over channels if multiple channels present - WLOG, averaging before or after resampling is equivalent
        data1D = np.mean(data2D, axis=0)

        # Test if there are at least 2 timesteps in this trial
        nTimeThis = len(data1D)
        if nTimeThis <= 1:
            raise ValueError("Trying to resample a degenerate time series of length", nTimeThis)

        # Resample data to target number of steps using Gaussian Kernel smoothening
        xSrc = np.linspace(0, 1, nTimeThis)
        rezLst += [resample(xSrc, data1D, xTrg, paramResample)]

    # Also average over trials
    return np.mean(rezLst, axis=0)