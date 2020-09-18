import numpy as np

from mesostat.stat.stat import rand_bool_perm


def _get_nan_rows(x):
    if not np.issubdtype(x.dtype, np.number):
        return np.zeros(x.shape).astype(bool)
    elif x.ndim <= 1:
        return np.isnan(x)
    else:
        return np.any(np.isnan(x), axis=tuple(list(range(1, x.ndim))))


# Drop all rows for which there is at least one NAN value
def drop_nan_rows(lst):
    nonNanLst = [~_get_nan_rows(x) for x in lst]
    goodIdxs = np.logical_and.reduce(nonNanLst)
    return [x[goodIdxs] for x in lst]


# CrossValidation: Split data into training and test sets along the first dimension
def split_train_test(dataLst, fracTest):
    n = len(dataLst[0])
    nTest = int(fracTest * n)
    testIdxs = rand_bool_perm(nTest, n)
    trainLst = [data[~testIdxs] for data in dataLst]
    testLst = [data[testIdxs] for data in dataLst]
    return trainLst + testLst


# https://en.wikipedia.org/wiki/Cohen%27s_kappa
def cohen_kappa(cm):
    cmNorm = cm / np.sum(cm)
    p0 = np.sum(cmNorm.diag())
    pe = np.sum([np.sum(cmNorm[i]) * np.sum(cmNorm[:, i]) for i in range(len(cm))])
    return (p0 - pe) / (1 - pe)



