import numpy as np
from time import time
import multiprocessing, pathos

# A class that switches between serial and parallel mappers
# Also deletes parallel mapper when class is deleted
class GenericMapper():
    def __init__(self, serial, nCore=None, verbose=True):
        self.serial = serial
        self.verbose = verbose
        self.pid = multiprocessing.current_process().pid

        if serial:
            self.nCore = 1
            self.map_func = lambda f,x: list(map(f, x))
        else:
            self.nCore = nCore if nCore is not None else pathos.multiprocessing.cpu_count() - 1
            self.pool = pathos.multiprocessing.ProcessingPool(self.nCore)
            # self.pool = multiprocessing.Pool(self.nCore)
            self.map_func = lambda f, x: self.pool.map(f, x)

    # def __del__(self):
    #     if not self.serial:
    #         self.pool.close()
    #         self.pool.join()

    def map(self, f, x):
        if self.verbose:
            t1 = time()
            print("----Root process", self.pid, "started task on", self.nCore, "cores----")

        rez = self.map_func(f, x)

        if self.verbose:
            print("----Root process", self.pid, "finished task. Time taken", np.round(time() - t1, 3), "seconds")

        return rez

    def mapMultiArg(self, f, x):
        # Ugly intermediate function to unpack tuple
        f_proxy = lambda task: f(*task)
        return self.map(f_proxy, x)
