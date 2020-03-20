import numpy as np

from mesostat.utils.arrays import numpy_transpose_byorder, test_have_dim

# Compute the total correlation, normalized by number of processes/channels
def total_correlation(h1D, hND, dimOrder1D, dimOrderND):
    # Both should have the same axis except 1D will have an extra processes axis
    assert len(dimOrder1D) == len(dimOrderND)+1, "Orders must be comparable"
    for d in dimOrderND:
        assert d in dimOrder1D, "Orders must be comparable"

    # Find processes axis
    idxProcess = dimOrder1D.index("p")
    nChannel = dimOrder1D.shape[idxProcess]

    # Average over processes axis
    h1Davg = np.mean(h1D, axis=idxProcess)

    # Remove processes axis from dimOrder
    dimOrder1DPost = dimOrder1D[:idxProcess] + dimOrder1D[idxProcess+1:]

    # Transpose both vectors to match
    h1Davgtr = numpy_transpose_byorder(h1Davg, dimOrder1DPost, dimOrderND)

    # Subtract
    return h1Davgtr - hND / nChannel