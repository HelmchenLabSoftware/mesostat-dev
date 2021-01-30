import os
import h5py


def type_of_object(obj):
    if isinstance(obj, h5py.Dataset):
        return 'Dataset', obj.shape
    elif isinstance(obj, h5py.Group):
        return 'Group', None
    else:
        raise ValueError('Unexpected H5PY element type', type(obj))


def type_of_path(fname, h5path):
    if not os.path.isfile(fname):
        raise IOError(fname, 'not found')

    with h5py.File(fname, 'r') as h5file:
        if h5path in h5file:
            return type_of_object(h5file[h5path])
        else:
            return None
