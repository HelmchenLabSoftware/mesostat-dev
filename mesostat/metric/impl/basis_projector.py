import numpy as np
from scipy.special import legendre

class BasisProjector:
    def __init__(self, nStep, order=5):
        self.poly = [legendre(iOrd) for iOrd in range(order)]

        # Sample in midpoints in order to reduce finite sample error
        x = np.linspace(-1, 1, nStep, endpoint=False) + 1 / nStep
        rezLst = [np.polyval(p, x) for p in self.poly]
        nrmLst = np.sqrt([np.dot(v, v) for v in rezLst])
        self.polysample = np.array([rez / nrm for rez, nrm in zip(rezLst, nrmLst)])

    def project(self, data):
        return self.polysample.dot(data)
