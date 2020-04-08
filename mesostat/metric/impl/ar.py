import numpy as np


# Fit AR(1) coefficients to data
def fit(x, y):
    '''
    :param x: array of shape [nTrial, timeWindow] - past values of 1D variable over several time steps
    :param y: array of shape nTrial - future values of 1D variable
    :return: vector of AR coefficients
    '''

    v = y.dot(x)
    M = x.T.dot(x)
    return np.linalg.solve(M, v)


# Compute prediction
def predict(x, alpha):
    return x.dot(alpha)


# Compute relative prediction error
def rel_err(y, yhat):
    return np.linalg.norm(y - yhat) / np.linalg.norm(y)


# Estimate timescale under hypothesis of convolution with exponential kernel
def tau_exp(fps, alpha):
    dt = 1 / fps
    return -dt / np.log(alpha[-1])