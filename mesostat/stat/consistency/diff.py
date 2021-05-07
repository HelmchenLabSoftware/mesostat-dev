import numpy as np


# Relative mean absolute error between two variables. Calculated in percent, namely in [0, 1]
# Maximal error reached when variables mismatch or anti-corrlate, does not discriminate between the two
def rmae(x, y):
    d = np.sum(np.abs(x - y))
    norm1 = np.sum(np.abs(x))
    norm2 = np.sum(np.abs(y))
    return d / (norm1 + norm2)


# Relative mean absolute error between two variables. Calculated in percent, namely in [0, 1]
# Maximal error reached when variables anti-corrlate. If variables mismatch only, error is 0.5
def rmsq(x, y):
    d = np.sum((x - y)**2) / 2
    norm1 = np.sum(x**2)
    norm2 = np.sum(y**2)
    return np.sqrt(d / (norm1 + norm2))
