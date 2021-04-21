import numpy as np
from idtxl.bivariate_pid import BivariatePID
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data

from mesostat.utils.decorators import redirect_stdout


def _parse_channels(settings):
    if 'channels' in settings.keys():
        assert len(settings['channels']) == 3
        src = settings['channels'][:2]
        trg = settings['channels'][2]
    else:
        src = settings['src']
        trg = settings['trg']
    return src, int(trg)


@redirect_stdout
def bivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps', normalise=False)
    pid = BivariatePID()

    src, trg = _parse_channels(settings)
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=trg, sources=src)

    return np.array([
        rez.get_single_target(trg)['unq_s1'],
        rez.get_single_target(trg)['unq_s2'],
        rez.get_single_target(trg)['shd_s1_s2'],
        rez.get_single_target(trg)['syn_s1_s2']
    ])


@redirect_stdout
def multivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps')
    pid = MultivariatePID()

    src, trg = _parse_channels(settings)
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=trg, sources=src)

    return np.array([
        rez.get_single_target(trg)['avg'][((1,),)][2],
        rez.get_single_target(trg)['avg'][((2,),)][2],
        rez.get_single_target(trg)['avg'][((1,), (2,),)][2],
        rez.get_single_target(trg)['avg'][((1, 2,),)][2]
    ])
