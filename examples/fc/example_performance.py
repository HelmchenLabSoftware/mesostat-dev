from time import time
import numpy as np
from mesostat.metric.metric import MetricCalculator

dataShape = (100, 1000, 10)
dimOrd = "rsp"
data = np.random.uniform(0,1,dataShape)

def timeIt(f):
    t1 = time()
    f()
    return time() - t1

def task(mc):
    mc.set_data(data, dimOrd, timeWindow=20)
    rez = mc.metric3D("avg_entropy", "s", metricSettings={"max_lag" : 5})
    print(rez.shape)

def serial():
    task(MetricCalculator(serial=True))

def parallel(nCore):
    task(MetricCalculator(serial=False, nCore=nCore))

times = [timeIt(serial)]

nCoreMax = 4
for nCore in range(1, nCoreMax+1):
    times += [timeIt(lambda : parallel(nCore))]

print(times)
