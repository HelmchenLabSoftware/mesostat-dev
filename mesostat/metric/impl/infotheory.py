import numpy as np

def entropy_discrete_1D(p1d):
    return -np.sum([p * np.log(p) for p in p1d])