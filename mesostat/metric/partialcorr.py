import numpy as np
from scipy import stats, linalg

'''
TODO:
[] Nan-protection
[] RSP version
[] cross-version
[] basic version
'''


def fit_covariate(x, cov):
    coeffX = linalg.lstsq(cov, x)[0]
    return x - cov.dot(coeffX)


# def partial_corr(x, y, covarX, covarY=None):
#     if covarY is None:
#         covarY = covarX
#
#     # Fit covariates to the
#     coeffX = linalg.lstsq(covarX, x)[0]
#     coeffY = linalg.lstsq(covarY, y)[0]
#
#     xhat = x - covarX.dot(coeffX)
#     yhat = y - covarY.dot(coeffY)
#
#     return stats.pearsonr(xhat, yhat)[0]


def partial_autocorr(x1D, ext=None):
    return partial_crosscorr(x1D, x1D, ext=ext)


# https://en.wikipedia.org/wiki/Partial_autocorrelation_function
def partial_crosscorr(dataSP, ext=None):
    nTime, nCh = dataSP.shape
    assert nTime > 20
    assert ext >= 1

    if ext is None:
        ext = nTime

    rez = np.zeros((ext, nCh, nCh), dtype=np.nan)
    rez[0] = np.corrcoef(dataSP)
    rez[1] = np.corrcoef(dataSP[1:], dataSP[:-1])[nCh:, :nCh]

    for t in range(2, ext):
        dataPastSP = dataSP[:-t]
        dataNextSP = dataSP[t:]
        dataCovSSP = np.array([dataSP[i:-t+i] for i in range(1, t)])

        # xSh = x1D[t:]
        # ySh = y1D[:-t]
        # xCov = np.array([x1D[i:-t+i] for i in range(1, t)])
        # yCov = np.array([y1D[i:-t + i] for i in range(1, t)])
        # xHat = fit_covariate(xSh, xCov)
        # yHat = fit_covariate(ySh, yCov)
        # rez[t] = stats.pearsonr(x1D, y1D)[0]


def partial_crosscorr_3D(dataRSP):
    pass