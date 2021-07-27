import unittest
import numpy as np

import mesostat.utils.arrays as arrays


class TestUtilSignalFit(unittest.TestCase):

    def test_numpy_move_dimension(self):
        # 2D
        data2D = np.random.normal(0, 1, (10, 20))
        assert arrays.numpy_move_dimension(data2D, 0, 0).shape == (10, 20)
        assert arrays.numpy_move_dimension(data2D, 0, 1).shape == (20, 10)
        assert arrays.numpy_move_dimension(data2D, 1, 0).shape == (20, 10)
        assert arrays.numpy_move_dimension(data2D, 1, 1).shape == (10, 20)

        # 3D
        data3D = np.random.normal(0, 1, (10, 20, 30))
        assert arrays.numpy_move_dimension(data3D, 0, 0).shape == (10, 20, 30)
        assert arrays.numpy_move_dimension(data3D, 0, 1).shape == (20, 10, 30)
        assert arrays.numpy_move_dimension(data3D, 0, 2).shape == (20, 30, 10)
        assert arrays.numpy_move_dimension(data3D, 1, 0).shape == (20, 10, 30)
        assert arrays.numpy_move_dimension(data3D, 1, 1).shape == (10, 20, 30)
        assert arrays.numpy_move_dimension(data3D, 1, 2).shape == (10, 30, 20)
        assert arrays.numpy_move_dimension(data3D, 2, 0).shape == (30, 10, 20)
        assert arrays.numpy_move_dimension(data3D, 2, 1).shape == (10, 30, 20)
        assert arrays.numpy_move_dimension(data3D, 2, 2).shape == (10, 20, 30)

    def test_numpy_merge_dimensions(self):
        nX, nY, nZ = 10, 20, 30

        #3D
        dataABC = np.random.normal(0, 1, (nX, nY, nZ))
        dataAB_C = arrays.numpy_merge_dimensions(dataABC, 0, 2)
        dataA_BC = arrays.numpy_merge_dimensions(dataABC, 1, 3)
        np.testing.assert_equal(dataAB_C.shape, (nX * nY, nZ))
        np.testing.assert_equal(dataA_BC.shape, (nX, nY * nZ))


unittest.main()
