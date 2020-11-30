import numpy as np

from mesostat.metric.metric import MetricCalculator


nChannel = 5
nTrial = 20
nTime = 100

data = np.random.randint(0, 4, (nChannel, nTrial, nTime)).astype(int)

mc = MetricCalculator(serial=False, nCore=4)
mc.set_data(data, 'prs')

settings_estimator = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}

rez = mc.metric3D('BivariatePID', '', metricSettings={'settings_estimator' : settings_estimator, 'parallelSrc3D' : [1,2,3], 'trg' : 4})

print(rez.shape)
print(rez[:, :, 0])