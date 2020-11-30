import numpy as np
import mesostat.stat.resampling as resampling

'''
TODO:
[?] Existing library for permutation testing
[ ] Accelerate testing by early stopping
[ ] Check if standard non-parametric test for equal variances exists (so far looks like active area of research)
'''



# def percentile_twosided(fTrue, fTestArr, settings=None):
#     nSample = fTestArr.shape[0]
#     pvalL = np.max([np.mean(fTestArr <= fTrue), 1/nSample])
#     pvalR = np.max([np.mean(fTestArr >= fTrue), 1/nSample])
#
#     output = [pvalL, pvalR]
#     if settings is not None:
#         if "haveEffectSize" in settings and settings["haveEffectSize"]:
#             effSize = (fTrue - np.mean(fTestArr)) / np.std(fTestArr)
#             output += [effSize]
#         if "haveMeans" in settings and settings["haveMeans"]:
#             output += [fTrue, np.mean(fTestArr)]
#
#     return output


# Evaluate in which percentile of the test data does the true value lay
# If function is multivariate, independent test is performed on each variable
# pvalL - probability of getting a random value at least as small as true
# pvalR - probability of getting a random value at least as large as true
# Note: The test cannot guarantee a pValue below 1/nSample, so in case of zero matches the pValue is upper-bounded
def percentile_twosided(fTrue, fTestArr, settings=None):
    # Find fraction of random results above and below true
    fracL = np.mean(fTestArr - fTrue <= 0, axis=0)
    fracR = np.mean(fTestArr - fTrue >= 0, axis=0)

    # Clip minimal possible p-value of this test to 1/N
    nSample = fTestArr.shape[0]
    pvalL = np.clip(fracL, 1/nSample, None)
    pvalR = np.clip(fracR, 1/nSample, None)

    output = [pvalL, pvalR]
    if settings is not None:
        if "haveEffectSize" in settings and settings["haveEffectSize"]:
            effSize = (fTrue - np.mean(fTestArr, axis=0)) / np.std(fTestArr, axis=0)
            output += [effSize]
        if "haveMeans" in settings and settings["haveMeans"]:
            output += [fTrue, np.mean(fTestArr, axis=0)]

    return np.array(output)


def perm_test_resample(f, x, nSample, permAxis=0, iterAxis=None, sampleFunction="permutation"):
    fSampleAxis = lambda x: resampling.sample(x, sampleFunction, permAxis=permAxis, iterAxis=iterAxis)
    return resampling.resample_monad(f, x, fSampleAxis, nSample=nSample)


def perm_test(f, x, nSample, permAxis=0, iterAxis=None, sampleFunction="permutation", settings=None):
    fTrue = f(x)
    fTestArr = perm_test_resample(f, x, nSample, permAxis=permAxis, iterAxis=iterAxis, sampleFunction=sampleFunction)
    return percentile_twosided(fTrue, fTestArr, settings=settings)


# Test if relative order of X vs Y matters.
# Tests if values of f(x,y) are significantly different if y is permuted wrt x
def paired_test(f, x, y, nSample, permAxis=0, iterAxis=None, sampleFunction="permutation", settings=None):
    assert x.shape == y.shape
    fTrue = f(x, y)
    fSampleAxis = lambda x: resampling.sample(x, sampleFunction, permAxis=permAxis, iterAxis=iterAxis)
    fTestArr = resampling.resample_dyad_individual(f, x, y, fSampleAxis, nSample=nSample)
    return percentile_twosided(fTrue, fTestArr, settings=settings)


# Tests whether a certain function is significantly different for X as opposed to Y values
# The values of X and Y are resampled from the shared pool
def difference_test(f, x, y, nSample, permAxis=0, iterAxis=None, sampleFunction="resample", settings=None):
    assert len(x) > 1, "Must have more than 1 sample to permute"
    assert len(y) > 1, "Must have more than 1 sample to permute"

    fSampleAxis = lambda x: resampling.sample(x, sampleFunction, permAxis=permAxis, iterAxis=iterAxis)
    fTrue = f(x, y)
    fTestArr = resampling.resample_dyad_union(f, x, y, fSampleAxis, nSample=nSample)
    return percentile_twosided(fTrue, fTestArr, settings=settings)
