import os
import unittest
import h5py
from mesostat.utils import hdf5_helper


class TestMetricAutocorr(unittest.TestCase):

    def test_type_of_path(self):
        datasetShape = (100,)

        with h5py.File('test.h5', 'w') as f:
            f.create_group('cats')
            f['cats'].create_dataset('tom', datasetShape, int)

        self.assertEqual(hdf5_helper.type_of_path('test.h5', '/cats'),       ('Group', None))
        self.assertEqual(hdf5_helper.type_of_path('test.h5', '/tom'),        None)
        self.assertEqual(hdf5_helper.type_of_path('test.h5', '/cats/tom'),   ('Dataset', datasetShape))
        self.assertEqual(hdf5_helper.type_of_path('test.h5', '/cats/jerry'), None)

        os.remove('test.h5')


unittest.main()
