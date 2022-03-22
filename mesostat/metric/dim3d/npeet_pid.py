import numpy as np
import npeet.entropy_estimators as ee

from mesostat.utils.decorators import redirect_stdout
from mesostat.metric.dim3d.common import shuffle_target, parse_channels


def pid_barrett(x,y,z):
    miX_Z = ee.mi(x, z, k=5, base=2)
    miY_Z = ee.mi(y, z, k=5, base=2)
    xy = np.array([x,y]).T
    miXY_Z = ee.mi(xy, z, k=5, base=2)

    red = min(miX_Z, miY_Z)
    unqX = miX_Z - red
    unqY = miY_Z - red
    syn = miXY_Z - unqX - unqY - red
    return unqX, unqY, red, syn


def pid_barrett_3D(dataRPS: np.array, settings: dict):
    [src1, src2], trg = parse_channels(settings, dim=3)

    dataRPSsh = shuffle_target(dataRPS, trg, settings)
    dataSrc1 = dataRPSsh[:, src1].flatten()
    dataSrc2 = dataRPSsh[:, src2].flatten()
    dataTrg = dataRPSsh[:, trg].flatten()

    rez = pid_barrett(dataSrc1, dataSrc2, dataTrg)

    # Getting rid of negative and very low positive PID's.
    # Statistical tests behave unexplectedly - perhaps low values contaminated by roundoff errors?
    return np.array(list(rez))  #np.clip(rez, 1.0E-6, None)
