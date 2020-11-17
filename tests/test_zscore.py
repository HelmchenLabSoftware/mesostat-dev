import numpy as np

from mesostat.utils.signals import zscore

###########################
# Test 2D
###########################

print("Test 2D")
nChannel = 10
nTime = 100

muLst = np.random.uniform(-5, 5, nChannel)
s2Lst = np.random.uniform(1, 50, nChannel)

x2d = np.array([np.random.normal(mu, np.sqrt(s2), nTime) for mu, s2 in zip(muLst, s2Lst)])

print("means", np.mean(x2d, axis=1))
print("vars", np.var(x2d, axis=1))

xZ = zscore(x2d, axis=1)
print("meansZ", np.mean(xZ, axis=1))
print("varsZ", np.var(xZ, axis=1))

###########################
# Test 3D
###########################
print("Test 3D")

nTrial = 200
nChannel = 10
nTime = 100

muLst = np.random.uniform(-5, 5, nChannel)
s2Lst = np.random.uniform(1, 50, nChannel)

x3d = np.array([np.random.normal(mu, np.sqrt(s2), (nTime, nTrial)) for mu, s2 in zip(muLst, s2Lst)])

print('shape', x3d.shape)
print("means", np.mean(x3d, axis=(1, 2)))
print("vars", np.var(x3d, axis=(1, 2)))

xZ = zscore(x3d, axis=(1, 2))
print("meansZ", np.mean(xZ, axis=(1, 2)))
print("varsZ", np.var(xZ, axis=(1, 2)))

