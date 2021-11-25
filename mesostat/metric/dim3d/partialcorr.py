import numpy as np

from mesostat.metric.impl.linalg import residual_linear_fit
from mesostat.metric.dim3d.common import shuffle_target, parse_channels

'''
TODO:
[] Nan-protection
[+] RSP version
[] cross-version
[+] basic version
'''


def partial_corr(x, y, covar, eta=1.0E-6):
    xFit = residual_linear_fit(x, covar)
    yFit = residual_linear_fit(y, covar)


    # If covariate explains one or both of the variables perfectly or almost perfectly,
    #   then residual may be either zero or dominated by numerical errors that need not be random.
    #   In this case behaviour of the metric is unpredictable and unreliable.
    # Add noise of very low relative magnitude to destroy very small effects
    stdX = np.std(x)
    stdY = np.std(y)
    noiseStdX = stdX * eta if stdX > 0 else eta
    noiseStdY = stdY * eta if stdY > 0 else eta
    xFit += np.random.normal(0, noiseStdX, x.shape)
    yFit += np.random.normal(0, noiseStdY, y.shape)

    rez = np.corrcoef(xFit, yFit)[0, 1]

    if np.isnan(rez):
        raise ValueError("Sth Went wrong")

    return np.clip(rez, eta, None)  # Crop very small values


def partial_corr_3D(dataRPS: np.array, settings: dict):
    [src1, src2], trg = parse_channels(settings, dim=3)

    dataRPSsh = shuffle_target(dataRPS, trg, settings)
    dataSrc1 = dataRPSsh[:, src1].flatten()
    dataSrc2 = dataRPSsh[:, src2].flatten()
    dataTrg = dataRPSsh[:, trg].flatten()

    rez1 = partial_corr(dataSrc1, dataTrg, np.array([dataSrc2]))
    rez2 = partial_corr(dataSrc2, dataTrg, np.array([dataSrc1]))

    return np.array([rez1, rez2])




# def partial_autocorr(x1D, ext=None):
#     return partial_crosscorr(x1D, x1D, ext=ext)
#
#
# # https://en.wikipedia.org/wiki/Partial_autocorrelation_function
# def partial_crosscorr(dataSP, ext=None):
#     nTime, nCh = dataSP.shape
#     assert nTime > 20
#     assert ext >= 1
#
#     if ext is None:
#         ext = nTime
#
#     rez = np.zeros((ext, nCh, nCh), dtype=np.nan)
#     rez[0] = np.corrcoef(dataSP)
#     rez[1] = np.corrcoef(dataSP[1:], dataSP[:-1])[nCh:, :nCh]
#
#     for t in range(2, ext):
#         dataPastSP = dataSP[:-t]
#         dataNextSP = dataSP[t:]
#         dataCovSSP = np.array([dataSP[i:-t+i] for i in range(1, t)])
#
#         # xSh = x1D[t:]
#         # ySh = y1D[:-t]
#         # xCov = np.array([x1D[i:-t+i] for i in range(1, t)])
#         # yCov = np.array([y1D[i:-t + i] for i in range(1, t)])
#         # xHat = fit_covariate(xSh, xCov)
#         # yHat = fit_covariate(ySh, yCov)
#         # rez[t] = stats.pearsonr(x1D, y1D)[0]
#
#
# def partial_crosscorr_3D(dataRSP):
#     pass