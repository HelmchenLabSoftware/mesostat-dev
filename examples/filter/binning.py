import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.signals.resample import bin_data
from mesostat.utils.signals.filter import approx_decay_conv, zscore

x = np.random.normal(0,1,(2000, 3))

x = approx_decay_conv(x, 100, 1)
y = bin_data(x, 4, axis=1)
xz = zscore(x, axis=0)
yz = zscore(y, axis=0)


plt.figure()
plt.plot(xz[:, 1])
plt.plot(yz[:, 1])
plt.show()

