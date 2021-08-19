import unittest
import numpy as np

import mesostat.utils.arrays as arrays


class TestUtilSignalFit(unittest.TestCase):

    def test_slice_sorted(self):
        data = np.arange(50, 100)

        l, r = arrays.slice_sorted(data, [60, 80])
        self.assertEqual(l, 10)
        self.assertEqual(r, 31) # Note inclusive

    def test_perm_map_arr(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([5, 4, 2, 3, 1])
        c = np.array([4, 3, 1, 2, 0])
        rez = arrays.perm_map_arr(a, b)
        np.testing.assert_array_equal(c, rez)

    def test_perm_map_str(self):
        a = "abcde"
        b = "edbca"
        c = np.array([4, 3, 1, 2, 0])
        rez = arrays.perm_map_str(a, b)
        np.testing.assert_array_equal(c, rez)

    def test_unique_subtract(self):
        self.assertEqual(arrays.unique_subtract("abcde", "ad"), "bce")
        self.assertEqual(arrays.unique_subtract([1,2,3,4,5], [2,3]), [1,4,5])
        self.assertEqual(arrays.unique_subtract((1,2,3,4,5), (1,2)), (3,4,5))

    def test_assert_get_dim_idx(self):
        self.assertEqual(arrays.assert_get_dim_idx("prs", "r"), 1)

        with self.assertRaises(ValueError):
            arrays.assert_get_dim_idx("ps", "r")

    def test_list_assert_get_uniform_shape(self):
        # Test for empty lists
        with self.assertRaises(ValueError):
            arrays.list_assert_get_uniform_shape([], )

        # Test passes for uniform list
        sh = (5, 10)
        lst = [np.zeros(sh) for i in range(20)]
        self.assertEqual(arrays.list_assert_get_uniform_shape(lst), sh)

        # Test fails for non-uniform list
        with self.assertRaises(ValueError):
            lst += [np.zeros((2,3))]
            arrays.list_assert_get_uniform_shape(lst)

    ############################
    # Numpy arrays
    ############################

    def test_numpy_transpose_byorder(self):
        # Test data dim mismatch
        with self.assertRaises(ValueError):
            arrays.numpy_transpose_byorder(np.zeros((2,3,4)), 'ps', 'sp')

        # Test source target mismatch
        with self.assertRaises(ValueError):
            arrays.numpy_transpose_byorder(np.zeros((2,3,4)), 'ps', 'rs')
        with self.assertRaises(ValueError):
            arrays.numpy_transpose_byorder(np.zeros((2,3,4)), 'ps', 'psr')

        # Test permutation no augment
        rez = arrays.numpy_transpose_byorder(np.zeros((2, 3, 4)), 'psr', 'rps')
        self.assertEqual(rez.shape, (4, 2, 3))

        # Test augment bad input
        with self.assertRaises(ValueError):
            arrays.numpy_transpose_byorder(np.zeros((2,3,4)), 'ps', 'rs', augment=True)

        # Test permutation augment
        rez = arrays.numpy_transpose_byorder(np.zeros((2, 4)), 'rs', 'rps', augment=True)
        self.assertEqual(rez.shape, (2, 1, 4))

        rez = arrays.numpy_transpose_byorder(np.zeros((5)), 'r', 'prs', augment=True)
        self.assertEqual(rez.shape, (1, 5, 1))

    def test_numpy_shape_reduced_axes(self):
        self.assertEqual(arrays.numpy_shape_reduced_axes((10, 20, 30), None),   (1, 1, 1))
        self.assertEqual(arrays.numpy_shape_reduced_axes((10, 20, 30), 1),      (10, 1, 30))
        self.assertEqual(arrays.numpy_shape_reduced_axes((10, 20, 30), (1, 2)), (10, 1, 1))

    def test_numpy_add_empty_axes(self):
        self.assertEqual(arrays.numpy_add_empty_axes(np.zeros((10, 20)), [1, 3]).shape, (10, 1, 20, 1))
        self.assertEqual(arrays.numpy_add_empty_axes(np.zeros((10, 20)), [0, 1]).shape, (1, 1, 10, 20))
        self.assertEqual(arrays.numpy_add_empty_axes(np.zeros((10, 20)), [2, 3]).shape, (10, 20, 1, 1))
        self.assertEqual(arrays.numpy_add_empty_axes(np.zeros((10, 20)), [0, 2]).shape, (1, 10, 1, 20))

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

    def test_numpy_take_all(self):
        data = np.random.normal(0, 1, (3,4,5,6,7))
        rez = arrays.numpy_take_all(data, [1,2,4], [0, 0, 1])
        np.testing.assert_array_almost_equal(rez, data[:, 0, 0, :, 1])

    def test_numpy_nonelist_to_array(self):
        # Test fails for all NAN
        with self.assertRaises(ValueError):
            arrays.numpy_nonelist_to_array([None]*5)

        # Test fails for different sized arrays
        with self.assertRaises(ValueError):
            arrays.numpy_nonelist_to_array([np.zeros((5,6)), np.zeros((2, 3))])

        # Test easy success
        brr = [np.zeros((2,3)) for i in range(4)]
        rez = arrays.numpy_nonelist_to_array(brr)
        self.assertEqual(rez.shape, (4,2,3))

        # Test hard success - scalar
        rez = arrays.numpy_nonelist_to_array([np.array(i) for i in range(5)] + [None])
        np.testing.assert_array_equal(rez, [0,1,2,3,4,np.nan])

        # Test hard success - vector
        rez = arrays.numpy_nonelist_to_array([np.zeros((10,11)) for i in range(5)] + [None])
        self.assertEqual(rez.shape, (6, 10, 11))
        np.testing.assert_array_equal(rez[-1], np.full((10, 11), np.nan))


unittest.main()
