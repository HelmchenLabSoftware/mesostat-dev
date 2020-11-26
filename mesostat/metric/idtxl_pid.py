import numpy as np
import pandas as pd
from idtxl.bivariate_pid import BivariatePID
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data


def bivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps')
    pid = BivariatePID()
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=settings['trg'], sources=settings['src'])

    return np.array([
        rez.get_single_target(settings['trg'])['unq_s1'],
        rez.get_single_target(settings['trg'])['unq_s2'],
        rez.get_single_target(settings['trg'])['shd_s1_s2'],
        rez.get_single_target(settings['trg'])['syn_s1_s2']
    ])


def multivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps')
    pid = MultivariatePID()
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=settings['trg'], sources=settings['src'])

    return np.array([
        rez.get_single_target(settings['trg'])['avg'][((1,),)][2],
        rez.get_single_target(settings['trg'])['avg'][((2,),)][2],
        rez.get_single_target(settings['trg'])['avg'][((1,), (2,),)][2],
        rez.get_single_target(settings['trg'])['avg'][((1, 2,),)][2]
    ])
