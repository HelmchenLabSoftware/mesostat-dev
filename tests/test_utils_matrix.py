import numpy as np

import mesostat.utils.matrix as matrix

aaa = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, np.nan]])
print(matrix.drop_nan_rows(aaa))
print(matrix.drop_nan_cols(aaa))
