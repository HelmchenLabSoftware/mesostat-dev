import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def paired_comparison(x2D, y2D, debugPlot=False, dropFirst=None):
    assert x2D.shape[1] == y2D.shape[1], "must have same numebr of channels"
    nChannel = x2D.shape[1]

    expvarXX = np.sort(np.linalg.eigvals(np.cov(x2D.T)))[::-1]
    expvarYY = np.sort(np.linalg.eigvals(np.cov(y2D.T)))[::-1]

    pcaX = PCA(n_components=nChannel)
    pcaY = PCA(n_components=nChannel)
    pcaX.fit(x2D)
    pcaY.fit(y2D)

    rotX = pcaX.components_
    rotY = pcaY.components_

    covXY = np.cov(x2D.dot(rotY.T).T)
    covYX = np.cov(y2D.dot(rotX.T).T)

    expvarXY = np.diag(covXY).copy()
    expvarYX = np.diag(covYX).copy()

    # # Sensibility test 1: Covariance of (x transformed to PCA domain) should be a diagonal matrix with
    # # elements equal to the eigenvalues of covariance of x
    # covXX = np.cov(x2D.dot(rotX.T).T)
    # evals1 = np.diag(covXX)
    # evals2 = np.sort(np.linalg.eigvals(np.cov(x2D.T)))[::-1]
    # print(np.linalg.norm(evals1-evals2))

    # Sensibility test 2: The sum of explained variances should match before and after transform
    # print(np.sum(expvarXX) - np.sum(expvarXY))
    # print(np.sum(expvarYY) - np.sum(expvarYX))

    ratioXX = expvarXX / np.sum(expvarXX)
    ratioYY = expvarYY / np.sum(expvarYY)
    ratioXY = expvarXY / np.sum(expvarXY)
    ratioYX = expvarYX / np.sum(expvarYX)

    if debugPlot:
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].plot(ratioXX, label='XX')
        ax[0].plot(ratioYY, label='XY')
        ax[1].plot(ratioXY, label='YY')
        ax[1].plot(ratioYX, label='YX')
        ax[0].legend()
        ax[1].legend()
        plt.show()

    if dropFirst is not None:
        ratioXX = ratioXX[dropFirst:]
        ratioXY = ratioXY[dropFirst:]
        ratioYY = ratioYY[dropFirst:]
        ratioYX = ratioYX[dropFirst:]

    rezXY = 1 - np.sum(np.abs(ratioXX - ratioXY)) / 2
    rezYX = 1 - np.sum(np.abs(ratioYY - ratioYX)) / 2

    return rezXY, rezYX, ratioXX, ratioXY, ratioYY, ratioYX
