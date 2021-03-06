import numpy as np
from bisect import bisect_left


def gaussian(mu, s2):
    return np.exp(- mu**2 / (2 * s2))

# Combined estimate of mean and variance. Excludes nan values
def mu_std(x, axis=None):
    return np.nanmean(x, axis=axis), np.nanstd(x, axis=axis)


# Construct a 1D random array that has an exact number of ones and zeroes
def rand_bool_perm(nTrue, nTot):
    rv = np.random.uniform(0, 1, nTot)
    return rv < np.sort(rv)[nTrue]


# Compute log-likelihood from probabilities of independent observations
def log_likelihood(pLst, axis=0):
    return -2 * np.sum(np.log(pLst), axis=axis)


# Compute empirical CDF from
def continuous_empirical_CDF(sample):
    x = np.sort(sample)
    x = np.hstack([x[0], x])
    y = np.linspace(0, 1, len(x))
    return x, y


# Convert discrete PDF into CDF
def discrete_CDF(p):
    nCDF = len(p) + 1
    x = np.zeros(nCDF)
    for i in range(nCDF-1):
        x[i + 1] = x[i] + p[i]
    return x


# Construct CDF from discrete distribution
# Has same length as PDF, first probability non-zero, last probability is 1
def discrete_distr_to_cdf(distr):
    keysSorted = sorted(distr.keys())
    cdfP = np.array([distr[k] for k in keysSorted])
    for i in range(1, cdfP.size):
        cdfP[i] += cdfP[i - 1]
    return dict(zip(keysSorted, cdfP))


# Construct a PDF from a discrete sample. No binning - items must match exactly
def discrete_empirical_pdf_from_sample(sample):
    keys, vals = np.unique(sample, return_counts=True)
    vals = vals.astype(float) / np.sum(vals)
    return dict(zip(keys, vals))


# Draw N samples from a discrete probability distribution
def discrete_cdf_sample(cdf, nSample):
    cdfX = np.array(list(cdf.keys()))
    cdfP = np.array(list(cdf.values()))

    urand = np.random.uniform(0, 1, nSample)
    bisectIdxs = [bisect_left(cdfP, p) for p in urand]
    return cdfX[bisectIdxs]


# Convert 1D dataset to pmf
def convert_pmf(x, base):
    assert x.ndim == 1, "Only defined for 1D data"
    xnorm = x - base
    xnorm[xnorm < 0] = 0
    return xnorm / np.sum(xnorm)