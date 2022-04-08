import numpy as np

from mesostat.metric.impl.linalg import residual_linear_fit
from mesostat.metric.dim3d.common import shuffle_target, parse_channels


def r2(trg, srcLst):
    return 1 - np.var(residual_linear_fit(trg, np.array(srcLst))) / np.var(trg)


def pr2_quadratic_triplet_decomp_1D(dataSrc1, dataSrc2, dataTrg):
    '''
    :param dataSrc1:  First source channel, 1D array of floats
    :param dataSrc2:  Second source channel, 1D array of floats
    :param dataTrg:   Target channel, 1D array of floats
    :return:      4 relative explained variances: unique source 1, unique source 2, redundant, synergistic

    Partial R^2-based decomposition of triplet correlations onto unique, redundant and synergistic parts
    Unique and redundant terms are purely linear, synergistic term is quadratic.
    '''

    # Compute first two centered moments
    c1 = dataSrc1 - np.mean(dataSrc1)
    c2 = dataSrc2 - np.mean(dataSrc2)
    cTrg = dataTrg - np.mean(dataTrg)
    c12 = c1 * c2

    # Fit, compute ExpVariduals and related variances
    rev1 = np.clip(r2(cTrg, [c1]), 0, None)
    rev2 = np.clip(r2(cTrg, [c2]), 0, None)

    revFull = np.clip(r2(cTrg, [c1, c2, c12]), 0, None)
    revM1   = np.clip(r2(cTrg, [c2, c12]), 0, None)
    revM2   = np.clip(r2(cTrg, [c1, c12]), 0, None)
    revM12  = np.clip(r2(cTrg, [c1, c2]), 0, None)

    # Compute individual effects
    # Clip negative entries as they are meaningless
    unq1 = np.clip(revFull - revM1, 0, None)
    unq2 = np.clip(revFull - revM2, 0, None)
    syn12 = np.clip(revFull - revM12, 0, None)
    # red12 = np.clip(revFull - unq1 - unq2 - syn12, 0, None)

    # unq1 = revM12 - rev2
    # unq2 = revM12 - rev1
    red12 = np.clip(rev1 + rev2 - revM12, 0, None)

    # print(revFull, revM1, revM2, revM12)
    # print(revM12, rev1, rev2)

    return np.array([unq1, unq2, red12, syn12])


def pr2_quadratic_triplet_decomp_3D(dataRPS: np.array, settings: dict):
    [src1, src2], trg = parse_channels(settings, dim=3)

    dataRPSsh = shuffle_target(dataRPS, trg, settings)
    dataSrc1 = dataRPSsh[:, src1].flatten()
    dataSrc2 = dataRPSsh[:, src2].flatten()
    dataTrg = dataRPSsh[:, trg].flatten()

    return pr2_quadratic_triplet_decomp_1D(dataSrc1, dataSrc2, dataTrg)
