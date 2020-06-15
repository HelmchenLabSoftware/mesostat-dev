import numpy as np

'''
MAR MODEL:
    x(t) = sum_i A(i)x(t-i) + B u(t)
'''

def unstack_factor(alpha, nHist):
    nCh = alpha.shape[0]
    return np.hstack([alpha[:, nCh * i:nCh * (i + 1)] for i in range(nHist)])


def predict(x, alpha):
    # return np.einsum('ai,ijk', alpha, x)
    return x.dot(alpha.T)


def fit_mle(x, y):
    '''
    :param x:  Parameters of shape [nTrial, nInputFeature]
    :param y:  Values of shape [nTrial, nOutputFeature]
    :return:   AR Matrix of shape [nInputFeature, nOutputFeature]
    '''

    # Construct linear system for transition matrices
    # M11 = np.einsum('ajk,bjk', x, y)
    # M12 = np.einsum('ajk,bjk', x, x)

    # Construct linear system for transition matrices
    # NOTE: In this form, trials and timesteps are concatenated, so there is no explicit trial dimension
    M11 = x.T.dot(y)
    M12 = x.T.dot(x)

    # Solve system
    alpha = np.linalg.inv(M12).dot(M11).T
    return alpha


# Compute L2 norm of the fit
def rel_err(y, yhat):
    return np.linalg.norm(y - yhat) / np.linalg.norm(y)