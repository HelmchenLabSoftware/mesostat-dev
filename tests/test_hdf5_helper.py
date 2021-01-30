import h5py
from mesostat.utils import hdf5_helper

with h5py.File('test.h5', 'w') as f:
    f.create_group('cats')
    f['cats'].create_dataset('tom', (100,), int)

print(hdf5_helper.type_of_path('test.h5', '/cats'))
print(hdf5_helper.type_of_path('test.h5', '/tom'))
print(hdf5_helper.type_of_path('test.h5', '/cats/tom'))
print(hdf5_helper.type_of_path('test.h5', '/cats/jerry'))
