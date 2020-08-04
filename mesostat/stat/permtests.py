import numpy as np
import mesostat.stat.resampling as resampling

'''
TODO:
[?] Existing library for permutation testing
[ ] Accelerate testing by early stopping
[ ] Check if standard non-parametric test for equal variances exists (so far looks like active area of research)
'''


# Evaluate in which percentile of the test data does the true value lay
# pvalL - probability of getting a random value at least as small as true
# pvalR - probability of getting a random value at least as large as true
# Note: The test cannot guarantee a pValue below 1/nSample, so in case of zero matches the pValue is upper-bounded
def percentile_twosided(fTrue, fTestArr):
    nSample = fTestArr.shape[0]
    pvalL = np.max([np.mean(fTestArr <= fTrue), 1/nSample])
    pvalR = np.max([np.mean(fTestArr >= fTrue), 1/nSample])
    return pvalL, pvalR


# Test if relative order of X vs Y matters.
# Tests if values of f(x,y) are significantly different if y is permuted wrt x
def paired_test(f, x, y, nSample, sampleFunction="permutation"):
    assert x.shape == y.shape
    fTrue = f(x, y)
    fSample = resampling.get_sample_function(sampleFunction)
    fTestArr = resampling.resample_dyad_individual(f, x, y, fSample, nSample=nSample)
    return percentile_twosided(fTrue, fTestArr)


# Tests whether a certain function is significantly different for X as opposed to Y values
# The values of X and Y are resampled from the shared pool
def difference_test(f, x, y, nSample, sampleFunction="resample"):
    assert len(x) > 1, "Must have more than 1 sample to permute"
    assert len(y) > 1, "Must have more than 1 sample to permute"

    fSample = resampling.get_sample_function(sampleFunction)
    fTrue = f(x, y)
    fTestArr = resampling.resample_dyad_union(f, x, y, fSample, nSample=nSample)
    return percentile_twosided(fTrue, fTestArr)
