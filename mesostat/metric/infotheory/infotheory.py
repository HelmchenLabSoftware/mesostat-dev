import numpy as np


# Compute Shannon Entropy given 1D discrete PMF
def entropy_discrete_1D(p):
    return -np.sum(p.dot(np.log(p)))
