import numpy as np
import pandas as pd
from itertools import permutations

from idtxl.bivariate_pid import BivariatePID
from idtxl.data import Data


def bivariate_pid_3D(data):
    settings = {
        'settings_estimator': {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]},
        'src': [0, 1],
        'trg': 2
    }

    dataIDTxl = Data(data, dim_order='rps', normalise=False)
    pid = BivariatePID()

    rez = pid.analyse_single_target(settings=settings['settings_estimator'], data=dataIDTxl, target=settings['trg'], sources=settings['src'])

    return np.array([
        rez.get_single_target(settings['trg'])['unq_s1'],
        rez.get_single_target(settings['trg'])['unq_s2'],
        rez.get_single_target(settings['trg'])['shd_s1_s2'],
        rez.get_single_target(settings['trg'])['syn_s1_s2']
    ])

nChannel = 3
nTrial = 20
nTime = 100

dataRPS = np.random.randint(0, 4, (nTrial, nChannel, nTime)).astype(int)

####################################
# Test 1: Consistency across permutations
####################################

rez = []
pLst = list(permutations([0,1,2]))
for p in pLst:
    rez += [[p] + list(bivariate_pid_3D(dataRPS[:, p]))]

print(pd.DataFrame(rez, columns=['perm', 'U1', 'U2', 'Red', 'Syn']))