import unittest
import numpy as np

import mesostat.utils.signals.resample as resample


class TestUtilSignalFit(unittest.TestCase):

    def test_flat_reshape_reverse(self):
        a = np.random.randint(0, 4, (10, 20))
        b = a.flatten().reshape(a.shape)
        np.testing.assert_array_equal(a, b)

    def test_bin_1D(self):
        data = np.linspace(0, 10, 12)
        dataBin2D = resample.bin_data_1D(data, 2)
        dataBin3D = resample.bin_data_1D(data, 3)
        dataBin4D = resample.bin_data_1D(data, 4)

        np.testing.assert_array_equal(dataBin2D, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(dataBin3D, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(dataBin4D, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    def test_bin_ND(self):
        nX, nY, nZ = 10, 20, 30
        nBins = 4

        ## 2D
        data2D = np.random.normal(0,1, (nX, nY))
        bin2Dx = resample.bin_data(data2D, nBins=nBins, axis=0)
        bin2Dy = resample.bin_data(data2D, nBins=nBins, axis=1)

        for i in range(nX):
            np.testing.assert_array_equal(bin2Dx[i], resample.bin_data_1D(data2D[i], nBins=nBins))
        for j in range(nY):
            np.testing.assert_array_equal(bin2Dy[:,j], resample.bin_data_1D(data2D[:, j], nBins=nBins))

        ## 3D
        data3D = np.random.normal(0,1, (nX, nY, nZ))
        bin3Dx = resample.bin_data(data3D, nBins=nBins, axis=0)
        bin3Dy = resample.bin_data(data3D, nBins=nBins, axis=1)
        bin3Dz = resample.bin_data(data3D, nBins=nBins, axis=2)


unittest.main()
