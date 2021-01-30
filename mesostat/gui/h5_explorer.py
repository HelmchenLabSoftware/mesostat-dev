import numpy as np
import h5py
import pandas as pd

from mesostat.utils.hdf5_helper import type_of_object

class H5Explorer:
    def __init__(self, fname):
        self.fname = fname
        self.innerPath = []
        self.h5file = h5py.File(fname, 'r')

    # def __del__(self):
    #     self.h5file.close()

    def move_child(self, key):
        obj = self.object_current()
        if key not in obj.keys():
            raise ValueError(key, 'is not an element of', self.get_inner_path())
        elif not isinstance(obj[key], h5py.Group):
            return -1, 'Child element ' + key + ' is not a group. Request ignored'
        else:
            self.innerPath += [key]
            return 0, 'Success'

    def move_parent(self):
        if len(self.innerPath) == 0:
            return -1, 'Already at root. Request ignored'
        else:
            self.innerPath = self.innerPath[:-1]
            return 0, 'Success'

    def get_inner_path(self):
        return '/' + '/'.join(self.innerPath)

    def object_by_path(self, innerPath):
        return self.h5file[innerPath]

    def contents_by_path(self, path):
        obj = self.object_by_path(path)
        labels = list(obj.keys())
        typesShapes = [type_of_object(obj[label]) for label in labels]
        dfDict = {'label': labels, 'type': [e[0] for e in typesShapes], 'shape' : [e[1] for e in typesShapes]}
        return pd.DataFrame(dfDict).sort_values(by=['type'], ascending=False)

    def object_current(self):
        return self.object_by_path(self.get_inner_path())

    def contents_current(self):
        return self.contents_by_path(self.get_inner_path())

    def read_content_keys(self, keys):
        obj = self.object_current()
        return [np.copy(obj[k]) for k in keys]
