import os, sys
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt

from mesostat.metric.metric import MetricCalculator


def generate_data(nTrial, nData, isSelfPredictive, isLinear, isShifted):
    dataShape = (nTrial, nData)

    if isSelfPredictive:
        src = np.outer(np.ones(nTrial), np.linspace(0, 1, nData))
    else:
        src = np.random.normal(0, 1, dataShape)

    if isLinear:
        x = src
        y = 1 - 2 * src
    else:
        x = np.cos(2 * np.pi * src)
        y = np.sin(2 * np.pi * src)

    x += 0.2 * np.random.normal(0, 1, dataShape)
    y += 0.2 * np.random.normal(0, 1, dataShape)

    if isShifted:
        y = np.hstack(([y[-1]], y[:-1]))

    return np.array([x, y])


def write_phase_space_fig(data, path, dataName):
    plt.figure()
    plt.plot(data[0].flatten(), data[1].flatten(), '.')
    plt.savefig(os.path.join(path, "data_" + dataName + ".png"))
    plt.close()


def write_fc_lag_p_fig(rez, path, rezKey):
    # Plot results
    fig, ax = plt.subplots()
    ax.imshow(rez)

    print(rezKey)
    outNameBare = "_".join([k for k in rezKey if k is not None])
    plt.savefig(os.path.join(path, outNameBare + ".png"))
    plt.close()


########################
# Generic parameters
########################
outpath = "tmp_imgs"

nTime = 100
nTrial = 10
nCoreArr = np.arange(1,5)

########################
# Sweep parameters
########################

algDict = {
    "Metric"       : ["crosscorr", "crosscorr", "BivariateMI", "BivariateMI", "BivariateMI"],
    "Library"      : ["corr", "spr", None, None, None, None],
    "CMI"          : [None, None, "OpenCLKraskovCMI", "JidtGaussianCMI", "JidtKraskovCMI"],
}

excludeCMI = ["OpenCLKraskovCMI"]

timesDict = {}
ptests = []
for dataName in ["Linear", "Circular"]:
    data = generate_data(nTrial, nTime, False, dataName == "Linear", False)

    write_phase_space_fig(data, outpath, dataName)

    for metricName, estimator, cmi in zip(*algDict.values()):
        # Get label
        taskKey = (dataName, metricName, estimator, cmi)

        if cmi not in excludeCMI:
            print("computing", taskKey)

            # Get settings
            metricSettings = {
                'lag': 1,
                'min_lag_sources': 1,
                'max_lag_sources': 1,
                "cmi_estimator": cmi,
                'est': estimator,
                'parallelTrg': True
            }

            # Compute performance metric
            timesDict[taskKey] = []

            for nCore in nCoreArr:
                tStart = time()
                mc = MetricCalculator(nCore=nCore)
                mc.set_data(data, 'prs', zscoreDim="rs")
                rez = mc.metric3D(metricName, "", metricSettings=metricSettings)
                if metricName != "crosscorr":
                    rez = rez[0]

                timesDict[taskKey] += [time() - tStart]

                write_fc_lag_p_fig(rez, outpath, taskKey)

            print("Sleeping...")
            sleep(1)


print("Off diagonal p-values")
print(np.array(ptests))

plt.figure()
plt.title("Strong scaling")
for k,v in timesDict.items():
    plt.semilogy(nCoreArr, v, label=str(k))

plt.xlabel("nCores")
plt.ylabel("runtime, seconds")
plt.legend()
plt.show()