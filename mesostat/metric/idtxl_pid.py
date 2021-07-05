import numpy as np
from idtxl.bivariate_pid import BivariatePID
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data

from mesostat.utils.decorators import redirect_stdout


def _parse_channels(settings, dim):
    if 'channels' in settings.keys():
        assert len(settings['channels']) == dim
        src = settings['channels'][:-1]
        trg = settings['channels'][-1]
    else:
        src = settings['src']
        trg = settings['trg']
    return src, int(trg)


def bivariate_pid_key():
    return [
        'unq_s1',
        'unq_s2',
        'shd_s1_s2',
        'syn_s1_s2'
    ]


# Stacking output into numpy array for efficiency
# Making sure that output always has the same order
def multivariate_pid_key(dim):
    if dim == 3:
        return [
            ((1,),),            # Unq1
            ((2,),),            # Unq2
            ((1,), (2,),),      # Shd12
            ((1, 2,),)          # Syn12
        ]
    elif dim == 4:
        return [
            ((1,),),                    # Unq_1
            ((2,),),                    # Unq_2
            ((3,),),                    # Unq_3
            ((1,), (2,)),               # Shd_12
            ((1,), (3,)),               # Shd_13
            ((2,), (3,)),               # Shd_23
            ((1,), (2,), (3,)),         # Shd_123
            ((1, 2),),                  # Syn_12
            ((1, 3),),                  # Syn_13
            ((2, 3),),                  # Syn_23
            ((1, 2, 3),),               # Syn_123
            ((1,), (2, 3)),             # {1}{23}
            ((2,), (1, 3)),             # {2}{13}
            ((3,), (1, 2)),             # {3}{12}
            ((1, 2), (1, 3)),           # {12}{13}
            ((1, 2), (2, 3)),           # {12}{23}
            ((1, 3), (2, 3)),           # {13}{23}
            ((1, 2), (1, 3), (2, 3))    # {12}{13}{23}
        ]
    else:
        raise ValueError('Unexpected dimension', dim)


@redirect_stdout
def bivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps', normalise=False)
    pid = BivariatePID()

    src, trg = _parse_channels(settings, dim=3)
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=trg, sources=src)

    return np.array([rez.get_single_target(trg)[k] for k in bivariate_pid_key()])


@redirect_stdout
def multivariate_pid_3D(data, settings):
    dataIDTxl = Data(data, dim_order='rps')
    pid = MultivariatePID()

    src, trg = _parse_channels(settings, dim=3)
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=trg, sources=src)

    return np.array([rez.get_single_target(trg)[k] for k in multivariate_pid_key(dim=3)])


@redirect_stdout
def multivariate_pid_4D(data, settings):
    dataIDTxl = Data(data, dim_order='rps')
    pid = MultivariatePID()

    src, trg = _parse_channels(settings, dim=4)
    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=trg, sources=src)

    return np.array([rez.get_single_target(trg)[k] for k in multivariate_pid_key(dim=4)])
