# Import classes
import numpy as np
from idtxl.multivariate_pid import MultivariatePID
from idtxl.data import Data

dataOrig = np.random.randint(0, 4, (1209, 27, 1))

data = Data(dataOrig, 'rps', normalise=False)

pid = MultivariatePID()
settings_SxPID = {'pid_estimator': 'SxPID', 'lags_pid': [0, 0, 0]}
results_SxPID = pid.analyse_single_target(settings=settings_SxPID, data=data, target=3, sources=(0, 1, 2))
print(results_SxPID.get_single_target(3)['avg'])
