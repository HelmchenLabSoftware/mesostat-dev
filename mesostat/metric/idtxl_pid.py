import numpy as np
from idtxl.bivariate_pid import BivariatePID
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data

from mesostat.utils.decorators import redirect_stdout


@redirect_stdout
def bivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps', normalise=False)
    pid = BivariatePID()

    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=settings['trg'], sources=settings['src'])

    return np.array([
        rez.get_single_target(settings['trg'])['unq_s1'],
        rez.get_single_target(settings['trg'])['unq_s2'],
        rez.get_single_target(settings['trg'])['shd_s1_s2'],
        rez.get_single_target(settings['trg'])['syn_s1_s2']
    ])


@redirect_stdout
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
