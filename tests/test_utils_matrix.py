import unittest
import numpy as np

import mesostat.utils.matrix as matrix


class TestUtilIterSweep(unittest.TestCase):

    def test_drop_chanels(self):
        mat5 = np.arange(25).reshape((5, 5))
        mDrop = matrix.drop_channels(mat5, [1,2, 4])
        np.testing.assert_array_equal(mDrop, np.array([[0, 3], [15, 18]]))

    def test_drop_rows_cols(self):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, np.nan]])
        arrDropRows = matrix.drop_nan_rows(arr)
        arrDropCols = matrix.drop_nan_cols(arr)

        np.testing.assert_array_equal(arrDropRows, np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        np.testing.assert_array_equal(arrDropCols, np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]]))

    def test_paired_diff(self):
        diff = matrix.pairwise_differences([1, 3, 4])
        rez = np.array([[0, 2, 3], [-2, 0, 1], [-3, -1, 0]])
        np.testing.assert_array_equal(diff, rez)

    def test_copy_triangle_symmetric(self):
        M1 = np.array([[1,np.nan, np.nan], [4,5,np.nan], [7,8,9]])

        M1symU = matrix.matrix_copy_triangle_symmetric(M1, source='U')
        M1symL = matrix.matrix_copy_triangle_symmetric(M1, source='L')

        np.testing.assert_array_equal(M1symU, np.array([[1, np.nan, np.nan], [np.nan, 5, np.nan], [np.nan, np.nan, 9]]))
        np.testing.assert_array_equal(M1symL, np.array([[1, 4, 7], [4, 5, 8], [7, 8, 9]]))


unittest.main()