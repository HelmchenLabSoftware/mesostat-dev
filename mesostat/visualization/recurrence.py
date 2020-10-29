import numpy as np
import matplotlib.pyplot as plt
import ipywidgets

from mesostat.utils.opencv_helper import cvWriter
from mesostat.utils.arrays import numpy_merge_dimensions
from sklearn.decomposition import PCA


def distance_matrix(data):
    nDim, nTime = data.shape
    dataExtr = np.repeat(data[..., None], nTime, axis=2)
    delta = dataExtr - dataExtr.transpose((0,2,1))
    return np.linalg.norm(delta, axis=0)


def decorrelate(data):
    # Leading dimension must be channels
    if data.ndim > 2:
        dataEff = numpy_merge_dimensions(data, 1, data.ndim+1)
    else:
        dataEff = data

    pca = PCA(n_components=48)
    rez = pca.fit_transform(dataEff.T)

    print(rez.shape)

    rez /= np.std(rez, axis=0)
    return rez.T.reshape(data.shape)


class RecurrencePlot:
    def __init__(self, data, w0):
        self.nPoint, self.nTime = data.shape

        # Compute the recurrence plots with no threshold
        self.dist = distance_matrix(data)

        # Plot first
        picdata = self.get_plotdata(w0)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.pic = self.ax.imshow(picdata, cmap='binary', origin='lower')
        self.ax.set_title('Recurrence Plot', fontsize=16)

    def get_plotdata(self, w):
        thr = np.percentile(self.dist.flatten(), w)
        return self.dist <= thr

    def update(self, w):
        picnew = self.get_plotdata(w)
        self.pic.set_data(picnew)
        self.fig.canvas.draw_idle()

    def interact(self):
        ipywidgets.interact(self.update, w=(0, 100, 1))

    def write_video(self, fname, frate=10.0, codec='XVID'):
        frameDim = (self.nTime, self.nTime)

        with cvWriter(fname, frameDim, frate=frate, codec=codec) as writer:
            for w in range(101):
                data = self.get_plotdata(w)
                writer.write(data)


class RecurrencePlotMultitrial:
    def __init__(self, data3D, w0):
        self.nTrial, self.nPoint, self.nTime = data3D.shape
        binarize = lambda dist, w: dist <= np.percentile(dist, w)

        # Compute the recurrence plots with no threshold
        self.rezMat = np.zeros((101, self.nTime, self.nTime), dtype=float)
        for data in data3D:
            dist = distance_matrix(data)
            for w in range(101):
                self.rezMat[w] += binarize(dist, w).astype(float)

        self.rezMat /= self.nTrial

        # Plot first
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.pic = self.ax.imshow(self.rezMat[w0], cmap='viridis', origin='lower', vmin=0, vmax=1)
        self.ax.set_title('Recurrence Plot', fontsize=16)

    def update(self, w):
        self.pic.set_data(self.rezMat[w])
        self.fig.canvas.draw_idle()

    def interact(self):
        ipywidgets.interact(self.update, w=(0, 100, 1))

    def write_video(self, fname, frate=10.0, codec='XVID'):
        frameDim = (self.nTime, self.nTime)

        with cvWriter(fname, frameDim, frate=frate, codec=codec) as writer:
            for w in range(101):
                writer.write(self.rezMat[w])
