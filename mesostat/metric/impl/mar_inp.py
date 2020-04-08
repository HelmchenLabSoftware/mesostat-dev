import numpy as np
from mesostat.metric.impl.mar import unstack_factor, rel_err


def predict(x, alpha, u, beta):
    # return np.einsum('ai,ijk', alpha, x) + np.einsum('ai,ijk', beta, u)
    return x.dot(alpha.T) + u.dot(beta.T)


def fit_mle(x, y, u):
    # # Construct linear system for transition matrices
    # M11 = np.einsum('ajk,bjk', x, y)
    # M12 = np.einsum('ajk,bjk', x, x)
    # M13 = np.einsum('ajk,bjk', x, u)
    # M21 = np.einsum('ajk,bjk', u, y)
    # M22 = M13.T  #np.einsum('ajk,bjk', u, x)
    # M23 = np.einsum('ajk,bjk', u, u)

    # Construct linear system for transition matrices
    # NOTE: In this form, trials and timesteps are concatenated, so there is no explicit trial dimension
    M11 = x.T.dot(y)
    M12 = x.T.dot(x)
    M13 = x.T.dot(u)
    M21 = u.T.dot(y)
    M22 = M13.T
    M23 = u.T.dot(u)

    # Solve system
    M12INV = np.linalg.inv(M12)
    M23INV = np.linalg.inv(M23)
    TMP11 = M11 - M13.dot(M23INV.dot(M21))
    TMP12 = M12 - M13.dot(M23INV.dot(M22))
    TMP21 = M21 - M22.dot(M12INV.dot(M11))
    TMP22 = M23 - M22.dot(M12INV.dot(M13))

    alpha = np.linalg.inv(TMP12).dot(TMP11).T
    beta = np.linalg.inv(TMP22).dot(TMP21).T

    return alpha, beta