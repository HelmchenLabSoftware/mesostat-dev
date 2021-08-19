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

        rez1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        rez2 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        rez3 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

        np.testing.assert_array_equal(dataBin2D, rez1)
        np.testing.assert_array_equal(dataBin3D, rez2)
        np.testing.assert_array_equal(dataBin4D, rez3)

        np.testing.assert_array_equal(dataBin2D[::-1], rez1[::-1])
        np.testing.assert_array_equal(dataBin3D[::-1], rez2[::-1])
        np.testing.assert_array_equal(dataBin4D[::-1], rez3[::-1])

        # Test large array
        data = np.random.normal(0, 100, (830560,))
        resample.bin_data_1D(data, 4)

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

    # TODO: Test specific to performance
    def test_downsample_int(self):
        nX = 100
        x = np.arange(nX)

        data = np.random.normal(0, 1, (nX, 10, 20))
        x2, y2 = resample.downsample_int(x, data, 5)

        assert y2.shape == (20, 10, 20)

    # Test if different wrappers for downsampling do the same thing
    def test_resample_kernel(self):
        n1 = 100
        n2 = 70

        x1 = np.linspace(0, 1, n1)
        x2 = np.linspace(0, 1, n2)
        y1 = np.sin(4 * np.pi * x1)

        k1 = resample.resample_kernel(x1, x2)
        y2k1 = k1.dot(y1)

        k2 = resample.resample_kernel_same_interv(n1, n2)
        y2k2 = k2.dot(y1)

        y3 = resample.resample(x1, y1, x2, {'method': 'downsample', 'kind': 'kernel'})

        np.testing.assert_array_almost_equal(y3, y2k1)
        np.testing.assert_array_almost_equal(y3, y2k2)

unittest.main()
