import numpy as np

def discrete_mean_var(x, p):
    mu = p.dot(x)
    var = p.dot((x - mu)**2)
    return mu, var
