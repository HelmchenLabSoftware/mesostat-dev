import scipy


def accuracy(nHit, nMiss, nFA, nCR):
    return (nHit + nCR) / (nHit + nMiss + nFA + nCR)


def d_prime(nHit, nMiss, nFA, nCR):
    hitRate = nHit / (nHit + nMiss)
    faRate  = nFA  / (nFA + nCR)

    zHit = scipy.stats.norm.ppf(hitRate)
    zFA = scipy.stats.norm.ppf(faRate)

    #     hitRateAdjusted = (nHit + 0.5) /((nHit + 0.5) + nMiss + 1)
    #     faRateAdjusted  = (nFA  + 0.5) /((nFA  + 0.5) + nCR + 1)

    #     zHit = scipy.stats.norm.ppf(hitRateAdjusted)
    #     zFA = scipy.stats.norm.ppf(faRateAdjusted)

    return zHit - zFA