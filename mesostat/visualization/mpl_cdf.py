import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mesostat.stat.stat import continuous_empirical_CDF


def cdf_labeled(ax, dataLst, dataLabels, paramName, haveLog=False):
    for data, label in zip(dataLst, dataLabels):
        x, y = continuous_empirical_CDF(data)
        ax.plot(x, y, label=label)

    if haveLog:
        ax.set_xscale('log')
    ax.legend()
    ax.set_xlabel(paramName)
    ax.set_ylabel('CDF')