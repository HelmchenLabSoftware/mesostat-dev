import numpy as np
from scipy import stats, linalg

'''
TODO:
[] Nan-protection
[] RSP version
[] cross-version
[] basic version
'''


# Fit a set of covariates to 1D data, subtract, return residual
def residual_linear_fit(x1D, cov2D):
    coeffX = linalg.lstsq(cov2D, x1D)[0]
    return x1D - cov2D.dot(coeffX)


# To each channel, fit its past timesteps with lag lower than t
def residual_fit_past_2D(dataSP, t):
    nTime, nCh = dataSP.shape

    # Stack past data values over all time delays less than t
    dataSPpast = np.array([dataSP[k:-t + k] for k in range(t)])

    rezSP = np.zeros(dataSPpast.shape)
    for iCh in range(nCh):
        rezSP[:, iCh] = residual_linear_fit(dataSP[t:, iCh], dataSPpast[:, iCh])

    return rezSP


def partial_autocorr(x1D, ext=None):
    return partial_crosscorr(x1D[:, None], ext=ext)   # Interpret x1D as 2D array with only 1 channel


def partial_crosscorr(dataSP, ext):
    nTime, nCh = dataSP.shape
    assert nTime > 20
    assert ext >= 1

    if ext is None:
        ext = nTime

    rez = np.zeros((ext, nCh, nCh), dtype=np.nan)
    rez[0] = np.corrcoef(dataSP)
    rez[1] = np.corrcoef(dataSP[1:], dataSP[:-1])[nCh:, :nCh]
    for t in range(2, ext):

        

        # Fit past residuals to data
        dataSPhat = residual_fit_past_2D(dataSP, t - 1)

        # Compute correlation at the given lag
        rez[t] = np.corrcoef(dataSPhat[:-t], dataSPhat[t:])


def partial_crosscorr_3D(dataRSP):
    pass