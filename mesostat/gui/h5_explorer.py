import numpy as np
import h5py
import pandas as pd

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
        elemTypes = []
        shapes = []
        for label in labels:
            if isinstance(obj[label], h5py.Dataset):
                elemTypes += ['Dataset']
                shapes += [obj[label].shape]
            elif isinstance(obj[label], h5py.Group):
                elemTypes += ['Group']
                shapes += ['']
            else:
                raise ValueError('Unexpected H5PY element type', type(obj[label]))

        return pd.DataFrame({'label': labels, 'type': elemTypes, 'shape' : shapes}).sort_values(by=['type'], ascending=False)

    def object_current(self):
        return self.object_by_path(self.get_inner_path())

    def contents_current(self):
        return self.contents_by_path(self.get_inner_path())

    def read_content_keys(self, keys):
        obj = self.object_current()
        return [np.copy(obj[k]) for k in keys]
