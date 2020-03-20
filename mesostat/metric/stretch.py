import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
from mesostat.metric.impl.basis_projector import BasisProjector

def stretch_basis_projection(data, settings):
    test_have_dim("crosscorr", settings['dim_order'], "s")

    # Average all axis except samples
    axis = tuple([i for i,e in enumerate(settings['dim_order']) if e != "s"])
    dataAvg = np.nanmean(data, axis=axis)
    nSample = len(dataAvg)

    # Evaluate first few projections to Legendre basis
    bp = BasisProjector(nSample)
    return bp.project(dataAvg)


def stretch_basis_projection_non_uniform(dataLst, settings):
    return np.mean([stretch_basis_projection(data, settings) for data in dataLst], axis=0)

