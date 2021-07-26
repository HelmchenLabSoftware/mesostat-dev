import unittest
import numpy as np

import mesostat.utils.iterators.matrix as mat


class TestUtilIterMat(unittest.TestCase):

    def test_g_2D(self):
        lst = list(mat.iter_g_2D(3))
        self.assertEqual(lst, [(0, 1), (0, 2), (1, 2)])

    def test_gg_3D(self):
        lst = list(mat.iter_gg_3D(4))
        rezTmp = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        self.assertEqual(lst, rezTmp)

    def test_gn_3D(self):
        lst = list(mat.iter_gn_3D(4))
        rezTmp = [
            (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2),
            (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 3, 0), (2, 3, 1)]
        self.assertEqual(lst, rezTmp)

    def test_ggn_4D(self):
        lst = list(mat.iter_ggn_4D(4))
        rezTmp = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)]
        self.assertEqual(lst, rezTmp)


unittest.main()
