import numpy as np
import matplotlib.pyplot as plt


# Change the scale of x-axis to non-linear, to emphasize the smaller values of x compared to larger ones
def nonlinear_xaxis(ax, scale=1):
    def _f(x):
        y = x.copy()
        y[x > 0] = np.log(y[x > 0] / scale + 1)
        return y

    def _f_inv(y):
        x = y.copy()
        x[y > 0] = scale * (np.exp(x[y > 0]) - 1)
        return y

    # f = lambda x: np.log(x / scale + 1)
    # f_inv = lambda y: scale * (np.exp(y) - 1)
    ax.set_xscale('function', functions=(_f, _f_inv))

