import numpy as np

# Iterate over a multidigit number, where each digit has a different base
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


# Another implementation
# def uneven_base_arithmetic_iterator(bases):
#     ndim = len(bases)
#     idxs = np.zeros(ndim, dtype=int)
#
#     while True:
#         yield idxs
#
#         # Increase last digit by 1
#         idxIncr = ndim - 1
#         idxs[idxIncr] += 1
#
#         # Increase all other digits if they exceed their base
#         while idxs[idxIncr] == bases[idxIncr]:
#             if idxIncr == 0:
#                 return
#             else:
#                 idxs[idxIncr] = 0
#                 idxIncr -= 1
#                 idxs[idxIncr] += 1