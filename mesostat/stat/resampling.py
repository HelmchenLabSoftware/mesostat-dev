import numpy as np

# Resamples data along given axis using bootstrap method
def bootstrap(x, axis=0):
    nElemAxis = x.shape[axis]
    idxBoot = np.random.randint(0, nElemAxis, nElemAxis)
    return np.take(x, idxBoot, axis=axis)


# Resamples data along given axis using permutation method
def permutation(x, axis=0):
    nElemAxis = x.shape[axis]
    idxPerm = np.random.permutation(nElemAxis)
    return np.take(x, idxPerm, axis=axis)


# Resamples data along given axis using cycling method
def cycling(x, axis=0):
    nElemAxis = x.shape[axis]
    cyclePeriod = np.random.randint(nElemAxis)
    idxCycle = np.arange(nElemAxis)
    idxCycle = np.hstack((idxCycle[cyclePeriod:], idxCycle[:cyclePeriod]))
    return np.take(x, idxCycle, axis=axis)


def get_sample_function(name):
    if name == "permutation":
        return permutation
    elif name == "bootstrap":
        return bootstrap
    elif name == "cycling":
        return cycling
    else:
        raise ValueError("Unexpected resampling function", name)


def sample(x, methodName, permAxis=0, iterAxis=None):
    sampleFunc = get_sample_function(methodName)
    if iterAxis is None:
        return sampleFunc(x, axis=permAxis)
    else:
        # Apply sampling for every element of selected axis
        nElemAxis = x.shape[iterAxis]
        permAxisEff = permAxis if permAxis < iterAxis else permAxis-1   # Account for iterated array being smaller
        rezArr = np.array([sampleFunc(np.take(x, i, axis=iterAxis), axis=permAxisEff) for i in range(nElemAxis)])

        # Transpose to move the sampled axis back to where it was originally
        return np.rollaxis(rezArr, 0, iterAxis+1)


def resample_monad(f, x, sampleFunc, nSample=2000):
    '''
    :param f: Test statistic of 1 variable
    :param x: Data array of arbitrary dimension. First dimension has to be trials
    :param sampleFunc: Method to resample data
    :param nSample: Number of times to resample data
    :return: Array of test statistics calculated for resampled data
    '''

    return np.array([f(sampleFunc(x)) for i in range(nSample)])


# Dyad     ::: f is a function of two variables
# Union    ::: X and Y are resampled from their union
def resample_dyad_union(f,x,y, sampleFunc, nSample=2000):
    '''
    :param f: Test statistic of 2 variables
    :param x: Data array of arbitrary dimension. First dimension has to be trials
    :param y: Another dataset. All other shapes except first one has to be the same as for X
    :param sampleFunc: Method to resample data
    :param nSample: Number of times to resample data
    :return: Array of test statistics calculated for resampled data
    '''

    M, N = x.shape[0], y.shape[0]
    fmerged = lambda xy : f(xy[:M], xy[M:])
    return resample_monad(fmerged, np.concatenate([x, y], axis=0), sampleFunc, nSample=nSample)


# Dyad         ::: f is a function of two variables
# Individual   ::: X and Y are resampled by permuting their relative order
def resample_dyad_individual(f,x,y, sampleFunc, nSample=2000, resampleY=False):
    if resampleY:
        fEff = lambda x, y: f(sampleFunc(x), sampleFunc(y))
    else:
        # Note: There is no advantage in wasting time permuting both variables, permuting one is sufficient
        fEff = lambda x, y: f(sampleFunc(x), y)

    return np.array([fEff(x, y) for i in range(nSample)])

