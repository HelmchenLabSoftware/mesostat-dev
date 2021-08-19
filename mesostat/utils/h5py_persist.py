import os
import numpy as np
import h5py
import time


class h5persist:
    def __init__(self, fname, mode, waittime=1, nTry=-1, verbose=True):
        '''
        :param fname:     Path to the H5 file
        :param mode:      File open mode, e.g. 'r' or 'a'
        :param waittime:  Number of seconds to wait if file opening fails
        :param nTry:      Number of times to try open until give up and crash. Use -1 for infinity
        :param verbose:   Whether to write extra output

        Wrapper for H5PY, allowing to continuously retry open file after few seconds if file already open somewhere else
        Intended for parallel use to avoid collisions. Intended to be used with 'with' operator
        '''

        self.f = None
        self.fname = fname
        self.mode = mode
        self.waittime = waittime
        self.nTry = nTry
        self.verbose=verbose

    def __enter__(self):
        if (not os.path.isfile(self.fname)) and (self.mode == 'r'):
            raise IOError('Trying to open a non-existing file for reading', self.fname)

        nTry = 0
        while self.f is None:
            nTry += 1
            if (self.nTry > 0) and (nTry > self.nTry):
                raise ValueError('Failed to open file after', nTry, 'attempts out of', self.nTry)

            try:
                self.f = h5py.File(self.fname, self.mode)
            except:
                if self.verbose:
                    print(os.getpid(), "File ", self.fname, " locking failed, waiting...")
                time.sleep(self.waittime)

        if self.verbose:
            print(os.getpid(), 'Opened file', self.fname)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

        if self.verbose:
            print(os.getpid(), 'Closed file', self.fname)


if __name__ == "__main__":
    print("Testing h5py.lock. Try simultaneously on multiple processes to see effect")
    with h5persist('test.h5', 'a', waittime=1, verbose=True) as h5w:
        if 'catAge' in h5w.f.keys():
            print('Found existing cat', np.array(h5w.f['catAge']))
            del h5w.f['catAge']

        time.sleep(7)
        catAge = np.random.randint(0, 20)
        h5w.f['catAge'] = catAge
        print('New cat age', catAge)
