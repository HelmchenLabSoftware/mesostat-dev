# Import classes
import numpy as np
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data


def myfunction(data, settings):
    dataIDTxl = Data(data, dim_order='rps')
    pid = MultivariatePID()
    rez = pid.analyse_single_target(settings=settings,
                                    data=dataIDTxl, target=3, sources=(0,1,2))

    return rez.get_single_target(3)['avg']


dataOrig = np.random.randint(0, 4, (1209, 27, 1))
settings = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0, 0]}

print(myfunction(dataOrig, settings))
