import numpy as np

from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt


# NMF, assuming each neuron is just a scaled version of a global signal
class GlobalActivityEstimator:
    def __init__(self, data):
        self.nCell, self.nTrial, self.nTime = data.shape
        self.dataFlat = data.reshape((self.nCell, self.nTrial * self.nTime))
        self.dataFlat -= np.min(self.dataFlat)  # Add constant to ensure all cells have positive activity
        self.nParam = self.nCell + self.nTrial * self.nTime

    def _pack(self, a, b):
        return np.hstack([a, b])

    def _unpack(self, x):
        return x[:self.nCell], x[self.nCell:]

    def _L(self, a, b):
        return self.dataFlat - np.outer(a, b)

    def _L2(self, x):
        a, b = self._unpack(x)
        return np.linalg.norm(self._L(a, b)) ** 2

    def _jacL2(self, x):
        a, b = self._unpack(x)
        L = self._L(a, b)
        da = -2 * L.dot(b)
        db = -2 * a.dot(L)
        return self._pack(da, db)

    # Now attempt to fit baseline to the whole neuron set
    def fit(self):
        x0 = np.random.uniform(1, 5, self.nParam)

        nonNegativeBound = Bounds(np.zeros(self.nParam), np.full(self.nParam, np.inf))
        rez = minimize(self._L2, x0, method='L-BFGS-B', bounds=nonNegativeBound, jac=self._jacL2)

        if not rez.success:
            print(rez.success)
            raise ArithmeticError("Baseline Fitting did not succeed")

        self.aFit, self.bFit = self._unpack(rez.x)

    def plot_fit(self):
        def addHistCurve(ax, data):
            y, x = np.histogram(data, bins='auto', density=True)
            y[y == 0] = 1.0E-5
            ax.semilogy(x[:-1], y)

        dataFlatNoBase = self._L(self.aFit, self.bFit)

        fig, ax = plt.subplots(nrows=3, figsize=(10, 15))
        ax[0].set_title("Cell activity distribution")
        ax[1].set_title("Fitted baseline")
        ax[2].set_title("Cell activity distribution after baseline subtraction")
        ax[1].plot(self.bFit)
        for i in range(self.nCell):
            addHistCurve(ax[0], self.dataFlat[i])
            addHistCurve(ax[2], dataFlatNoBase[i])
        plt.show()