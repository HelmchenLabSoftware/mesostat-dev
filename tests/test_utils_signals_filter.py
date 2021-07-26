import unittest
import numpy as np

import mesostat.utils.signals.filter as filter
from mesostat.utils.arrays import numpy_merge_dimensions


class TestUtilSignalFilter(unittest.TestCase):

    def test_zscore(self):
        x = np.random.normal(0, 1, (10, 20, 30))
        xFx = filter.zscore(x, axis=0)
        xFy = filter.zscore(x, axis=1)
        xFz = filter.zscore(x, axis=2)
        xFxy = filter.zscore(x, axis=(0,1))
        xFxy2D = numpy_merge_dimensions(xFxy, 0, 2)

        np.testing.assert_almost_equal(np.std(xFx, axis=0), np.ones((20, 30)))
        np.testing.assert_almost_equal(np.std(xFy, axis=1), np.ones((10, 30)))
        np.testing.assert_almost_equal(np.std(xFz, axis=2), np.ones((10, 20)))
        np.testing.assert_almost_equal(np.std(xFxy2D, axis=0), np.ones((30,)))

    def test_zscore_dimord(self):
        x = np.random.normal(0, 1, (10, 20, 30))

        xTmp = filter.zscore_dim_ord(x, 'rps')
        xFx = filter.zscore_dim_ord(x, 'rps', 'r')
        xFy = filter.zscore_dim_ord(x, 'rps', 'p')
        xFz = filter.zscore_dim_ord(x, 'rps', 's')

        np.testing.assert_array_equal(xTmp, x)
        np.testing.assert_almost_equal(np.std(xFx, axis=0), np.ones((20, 30)))
        np.testing.assert_almost_equal(np.std(xFy, axis=1), np.ones((10, 30)))
        np.testing.assert_almost_equal(np.std(xFz, axis=2), np.ones((10, 20)))

    def test_zscore_lst(self):
        lst = [np.random.normal(0, 5, (10, 20)) for i in range(30)]
        rez = filter.zscore_list(lst)
        rez1D = np.hstack([data.flatten() for data in rez])
        np.testing.assert_almost_equal(np.std(rez1D), 1)

    # TODO: Improvement: Test effect, not only shape
    def test_drop_PCA(self):
        dataRP = np.random.normal(0, 1, (100, 10))
        rez = filter.drop_PCA(dataRP, nPCA=1)
        self.assertEqual(rez.shape, dataRP.shape)

    # TODO: Improvement: Test effect, not only shape
    def test_approx_decayconv(self):
        dataSPR = np.random.normal(0, 1, (100, 5, 10))
        rez = filter.approx_decay_conv(dataSPR, 0.5, 0.05)
        self.assertEqual(dataSPR.shape, rez.shape)


unittest.main()
