import numpy as np


def non_uniform_base_arithmetic_iterator(bases):
    assert np.min(bases) > 0, "all bases must be positive integers"
    nBase = len(bases)
    nNumber = int(np.prod(bases))
    v = np.zeros(nBase, dtype=int)

    yield v

    for iNumber in range(1, nNumber):
        iBase = nBase-1

        v[iBase] += 1
        while v[iBase] == bases[iBase]:
            v[iBase] = 0
            iBase -= 1
            v[iBase] += 1

        yield v
