from mesostat.utils.decorators import time_mem_1starg

@time_mem_1starg
def myfunc(x):
    return x**2

print(myfunc(10))