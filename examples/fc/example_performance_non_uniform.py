from time import time
import numpy as np
from mesostat.metric.metric_non_uniform import MetricCalculatorNonUniform

nTrial = 100
nChannel = 40
dataLst = [np.random.uniform(0,1, (nChannel, np.random.randint(500, 1000))) for iTrial in range(nTrial)]

def timeIt(f):
    t1 = time()
    f()
    return time() - t1

def task(mc):
    mc.set_data(dataLst)
    rez = mc.metric3D("avg_PI", "r", metricSettings={"max_lag" : 5})
    print(rez.shape)

def serial():
    task(MetricCalculatorNonUniform(serial=True))

def parallel(nCore):
    task(MetricCalculatorNonUniform(serial=False, nCore=nCore))

times = [timeIt(serial)]

nCoreMax = 4
for nCore in range(1, nCoreMax+1):
    times += [timeIt(lambda : parallel(nCore))]

print(times)
