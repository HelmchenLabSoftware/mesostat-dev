import unittest
import numpy as np

from mesostat.stat.combinatorics import log_comb, comb_fak


class TestCombinatorics(unittest.TestCase):

    # Computing rel. error of approximate number of combinations for small values
    def test_approx_exact(self):
        for i in range(20):
            for j in range(i):
                combExact = comb_fak(i, j)
                combApprox = np.exp(log_comb(i, j))
                np.testing.assert_almost_equal(combApprox, combExact)
