import numpy as np


##########################
# Deleting channels
##########################

# Drop rows and columns by index, assuming rows and columns have same indexing
def drop_channels(m, dropIdxs=None):
    if dropIdxs is None:
        return m
    else:
        mRez = np.copy(m)
        for iCh in sorted(dropIdxs)[::-1]:  # Sort in descending order, so index still correct after deletion
            mRez = np.concatenate([mRez[:iCh], mRez[iCh + 1:]], axis=0)
            mRez = np.concatenate([mRez[:, :iCh], mRez[:, iCh + 1:]], axis=1)
        return mRez


def drop_nan_rows(data2D):
    assert data2D.ndim == 2
    return data2D[~np.isnan(data2D).any(axis=1)]


def drop_nan_cols(data2D):
    assert data2D.ndim == 2
    return data2D[:, ~np.isnan(data2D).any(axis=0)]


##########################
# Constructing matrices
##########################


# Construct a matrix M_{ij} = a_i - a_j
def pairwise_differences(data1D):
    dataExtrudeX = np.outer(np.ones(len(data1D)), data1D)
    return dataExtrudeX - dataExtrudeX.T


# Construct a matrix where some upper off-diagonal is filled with ones
def setDiagU(n, sh, baseline=0):
    M = np.full((n, n), baseline)
    for i in range(n-sh):
        M[i + sh, i] = 1
    return M


##########################
# Indexing
##########################


# Return index of diagonal elements of a square matrix
def diag_idx(N):
    return np.eye(N, dtype=bool)


# Return index of off-diagonal elements of a square matrix
def offdiag_idx(N):
    return ~np.eye(N, dtype=bool)


# Return flat 1D array of matrix off-diagonal entries
def offdiag_1D(M):
    return M[offdiag_idx(M.shape[0])]

##########################
# Parts of matrix as 1D
##########################


def tril_1D(M):
    idxs = np.tril_indices(M.shape[0], k=-1)
    return M[idxs]


def triu_1D(M):
    idxs = np.triu_indices(M.shape[0], k=1)
    return M[idxs]


# Return flat 1D array of matrix off-diagonal entries, but only those which are not NAN
def offdiag_1D_nonnan(M):
    MOffDiag = offdiag_1D(M)
    return MOffDiag[~np.isnan(MOffDiag)]


##############################################
# Matrix Manipulation without shape change
##############################################


# Set diagonal to zero
def offdiag(M):
    MZeroDiag = np.copy(M)
    np.fill_diagonal(MZeroDiag, 0)
    return MZeroDiag


# Set diagonal to zero, then normalize
def offdiag_norm(M):
    MZeroDiag = offdiag(M)
    Mmax = np.max(np.abs(MZeroDiag))
    return MZeroDiag if Mmax == 0 else MZeroDiag / Mmax


# Makes square matrix symmetric by copying one of the off-diagonal triangles into another
def matrix_copy_triangle_symmetric(m, source='U'):
    if source == 'U':
        idxL = np.tril_indices(m.shape[0], -1)
        mNew = m.copy()
        mNew[idxL] = mNew.T[idxL]
        return mNew
    elif source == 'L':
        idxU = np.triu_indices(m.shape[0], 1)
        mNew = m.copy()
        mNew[idxU] = mNew.T[idxU]
        return mNew
    else:
        raise ValueError('Unexpected source', source)


# Make square matrix symmetric by reconciling incoming and outgoing links
def matrix_make_symm(M, either=True):
    if either:
        # Assuming undirected link has to be present at least in one direction
        if M.dtype == bool:
            return M or M.T
        else:
            return np.nanmean([M, M.T], axis=0) # nanmean sets result to NAN only if both are NAN
    else:
        # Assuming undirected link has to be present in both directions
        if M.dtype == bool:
            return M & M.T
        else:
            return np.mean([M, M.T], axis=0)   # Mean sets result to NAN if at least one entry is NAN
