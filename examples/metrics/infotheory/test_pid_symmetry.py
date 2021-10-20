import numpy as np
import pandas as pd

from mesostat.metric.dim3d.idtxl_pid import bivariate_pid_3D
from itertools import permutations


def eval_pid(data):
    settings = {
        'settings_estimator': {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]},
        'src': [0, 1],
        'trg': 2
    }
    return bivariate_pid_3D(data, settings)

nChannel = 3
nTrial = 20
nTime = 100

dataRPS = np.random.randint(0, 4, (nTrial, nChannel, nTime)).astype(int)

####################################
# Test 1: Consistency across runs
####################################

# rez = []
# for i in range(10):
#     rez += [eval_pid(dataRPS)]
# print(pd.DataFrame(rez, columns=['U1', 'U2', 'Red', 'Syn']))

####################################
# Test 2: Consistency across permutations
####################################

rez = []
pLst = list(permutations([0,1,2]))
for p in pLst:
    rez += [[p] + list(eval_pid(dataRPS[:, p]))]

print(pd.DataFrame(rez, columns=['perm', 'U1', 'U2', 'Red', 'Syn']))