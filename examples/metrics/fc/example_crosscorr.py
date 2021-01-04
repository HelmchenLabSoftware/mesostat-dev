# import standard libraries
import numpy as np
import matplotlib.pyplot as plt

from mesostat.metric.metric import MetricCalculator
from mesostat.visualization.mpl_matrix import plot_matrix

mc = MetricCalculator(serial=True, verbose=True)

'''
   Test 1:
     Generate random data, and shift it by fixed steps for each channel
     Expected outcomes:
     * If shift <= max_delay, corr ~ 1, delay = shift
     * If shift > max_delay, corr ~ 0, delay = rand
     * Delay is the same for all diagonals, because we compare essentially the same data, both cycled by the same amount
'''

nNode = 5
nData = 1000
data = np.zeros((nNode, nData))
data[0] = np.random.normal(0, 1, nData)
for i in range(1, nNode):
    data[i] = np.hstack((data[i-1][1:], data[i-1][0]))

mc.set_data(data, 'ps')
sweepSettings = {"lag" : [1,2]}
rezCorr = mc.metric3D("crosscorr", "", metricSettings={"estimator" : "corr"}, sweepSettings=sweepSettings)
rezSpr = mc.metric3D("crosscorr", "", metricSettings={"estimator" : "spr"}, sweepSettings=sweepSettings)
rezTot = np.concatenate([rezCorr, rezSpr], axis=0)

fig, ax = plot_matrix(rezTot, (2, 2),
    xlabels=["corr", "spr"],
    ylabels=["lag="+str(i) for i in sweepSettings["lag"]],
    lims = [[-1, 1]]*len(rezTot),
    title="Test 1: Channels are shifts of the same data",
    haveColorBar=True
)

plt.draw()


'''
   Test 2:
     Generate random data, all copies of each other, each following one a bit more noisy than prev
     Expected outcomes:
     * Correlation decreases with distance between nodes, as they are separated by more noise
     * Correlation should be approx the same for any two nodes given fixed distance between them
'''

nNode = 5
nData = 1000
alpha = 0.5

data = np.random.normal(0, 1, (nNode, nData))
for i in range(1, nNode):
    data[i] = data[i-1] * np.sqrt(1 - alpha) + np.random.normal(0, 1, nData) * np.sqrt(alpha)

mc.set_data(data, 'ps')
rezCorr = mc.metric3D("crosscorr", "", metricSettings={"estimator" : "corr", "lag" : 0})
rezSpr = mc.metric3D("crosscorr", "", metricSettings={"estimator" : "spr", "lag" : 0})
rezTot = np.array([rezCorr, rezSpr])

fig, ax = plot_matrix(rezTot, (1, 2),
    xlabels=["corr", "spr"],
    lims = [[-1, 1]]*len(rezTot),
    title="Test 2: Channels are same, but progressively more noisy",
    haveColorBar=True
)

plt.draw()


'''
   Test 3:
     Random data structured by trials. Two channels (0 -> 3) connected with lag 6, others unrelated
     Expected outcomes:
     * No structure, except for (0 -> 3) connection
'''

nNode = 5
maxLag = 6
lagTrue = 4
nData = maxLag + 1
nTrial = 200

data = np.random.normal(0, 1, (nTrial,nData,nNode))
data[:, lagTrue:, 0] = data[:, :-lagTrue, 3]

mc.set_data(data, 'rsp')
sweepSettings = {"lag" : np.arange(1, maxLag+1).astype(int)}
rezCorr = mc.metric3D("crosscorr", "", metricSettings={"estimator" : "corr"}, sweepSettings=sweepSettings)
rezSpr = mc.metric3D("crosscorr", "", metricSettings={"estimator" : "spr"}, sweepSettings=sweepSettings)
rezTot = np.concatenate([rezCorr, rezSpr], axis=0)

fig, ax = plot_matrix(rezTot, (2, len(sweepSettings['lag'])),
    ylabels=["corr", "spr"],
    xlabels=["lag="+str(i) for i in sweepSettings["lag"]],
    lims = [[-1, 1]]*len(rezTot),
    title=    "Test 3: Random trial-based cross-correlation",
    haveColorBar=True
)

plt.draw()
plt.show()