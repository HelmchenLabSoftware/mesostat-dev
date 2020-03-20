import sys
import numpy as np

from mesostat.utils.system import mem_now_as_str, bytes2str

def heavyFunc(i):
    x = np.random.normal(0,1, 10**8)
    print(i, mem_now_as_str(), bytes2str(sys.getsizeof(x)))
    return i


print(list(map(heavyFunc, np.arange(10))))