import unittest
import numpy as np

import mesostat.metric.dim1d.autocorr as ac


class TestMetricAutocorr(unittest.TestCase):

    def test_ac_1D(self):
        n = 100
        x = np.ones(n)
        rez = ac.autocorr_1D(x)
        rezTmp = (np.arange(1, n+1) / n)[::-1]

        self.assertEqual(rez.shape, (n,))
        np.testing.assert_array_equal(rez, rezTmp)

    def test_ac_3D(self):
        rsp = (10, 20, 30)
        x = np.random.normal(0.0, 1.0, rsp)
        rez = ac.autocorr_3D(x, {})
        self.assertEqual(rez.shape, (30,))

    def test_ac_3D_fewsamples(self):
        x = np.random.uniform(0, 1, (10, 10, 0))
        with self.assertRaises(ValueError):
            rez = ac.autocorr_3D(x, {})

    def test_ac_1D_trunc(self):
        lst = [
            np.random.normal(0, 1, (20, 33)),
            np.random.normal(0, 1, (20, 30)),
            np.random.normal(0, 1, (20, 31)),
            np.random.normal(0, 1, (20, 32))
        ]
        rez = ac.autocorr_trunc_3D(lst, {})
        self.assertEqual(rez.shape, (30,))

    def test_ac_d1_3D(self):
        x = np.random.normal(0, 1, (10,20,30))
        rez = ac.autocorr_d1_3D(x, {})
        self.assertEqual(rez.dtype, np.float64)

    def test_ac_d1_3D_non_uniform(self):
        lst = [
            np.random.normal(0, 1, (20, 33)),
            np.random.normal(0, 1, (20, 30)),
            np.random.normal(0, 1, (20, 31)),
            np.random.normal(0, 1, (20, 32))
        ]
        rez = ac.autocorr_d1_3D_non_uniform(lst, {})
        self.assertEqual(rez.dtype, np.float64)

    def test_ac_d1_3D_fewsamples(self):
        x = np.random.uniform(0, 1, (10, 10, 0))
        with self.assertRaises(ValueError):
            rez = ac.autocorr_d1_3D(x, {})

unittest.main()
