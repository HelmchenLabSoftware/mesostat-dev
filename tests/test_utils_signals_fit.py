import unittest
import numpy as np

import mesostat.utils.signals.fit as fit


class TestUtilSignalFit(unittest.TestCase):

    def test_polyfit_transform(self):
        x = np.arange(200) / 50
        y = np.sin(x)
        yHat = fit.polyfit_transform(x, y, ord=15)

        self.assertEqual(y.shape, yHat.shape)
        np.testing.assert_array_almost_equal(y, yHat, decimal=10)

    # TODO: Test nan-pathway
    def test_uniform_spline(self):
        x = np.arange(200) / 50
        y = np.sin(x)
        yHat = fit.natural_cubic_spline_fit_reg(x, y, dof=15)

        self.assertEqual(y.shape, yHat.shape)
        np.testing.assert_array_almost_equal(y, yHat, decimal=2)


unittest.main()
