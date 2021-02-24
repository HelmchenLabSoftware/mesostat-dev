import numpy as np


def discrete_mean_var(x, p):
    mu = p.dot(x)
    var = p.dot((x - mu)**2)
    return mu, var


# Compute indices of n largest values
# Works for arrays of arbitrary shape
def n_largest_indices(x, n):
    xFlat = x.flatten()
    ind = np.argpartition(xFlat, -n)[-n:]                        # Get the indices of n largest elements
    ind = ind[np.argsort(xFlat[ind])][::-1]                      # Sort the indices
    ind = [tuple(np.unravel_index(i, x.shape)) for i in ind]     # Convert index back to original shape
    return ind
