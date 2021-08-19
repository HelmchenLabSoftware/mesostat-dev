import unittest
import numpy as np

from mesostat.utils.algorithms import non_uniform_base_arithmetic_iterator


class TestNonUniformBaseArithmeticIterator(unittest.TestCase):

    # Construct an example number sequence and see if the test case produces it
    def test_3D(self):
        bases = [3, 2, 5]

        true = [[[(k,j,i) for i in range(bases[2])] for j in range(bases[1])] for k in range(bases[0])]
        true = np.array(true).reshape(np.prod(bases), len(bases))

        iter = non_uniform_base_arithmetic_iterator(bases)

        for it, t in zip(iter, true):
            assert np.array_equal(it, t)
