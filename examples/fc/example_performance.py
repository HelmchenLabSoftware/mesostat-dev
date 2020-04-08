import numpy as np
from mesostat.metric.metric import MetricCalculator

dataShape = (400, 100, 10)
dimOrd = "rsp"
data = np.random.uniform(0,1,dataShape)

mc = MetricCalculator(serial=True)
mc.set_data(data, dimOrd)
rez = mc.metric3D("mean", "rs")
print(rez.shape)
