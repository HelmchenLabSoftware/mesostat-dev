import numpy as np
import colorsys
from scipy import interpolate
from matplotlib.colors import ListedColormap


def random_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def custom_grad_cmap(colorArr):
    '''
    Generate a custom colormap given colors

    :param colorArr: array of shape [nPoint, 3] - an RGB color for each interpolation point
    :return: a matplotlib CMAP. Currently distributes the interpolation points evenly
    '''

    nStep = 256
    nPoint = len(colorArr)

    colorArrEff = colorArr if np.max(colorArr) <= 1 else colorArr / 255

    x = np.arange(nStep)
    x0 = np.linspace(0, nStep - 1, nPoint)

    vals = np.ones((nStep, 4))
    vals[:, 0] = interpolate.interp1d(x0, colorArrEff[:, 0], kind='linear')(x)
    vals[:, 1] = interpolate.interp1d(x0, colorArrEff[:, 1], kind='linear')(x)
    vals[:, 2] = interpolate.interp1d(x0, colorArrEff[:, 2], kind='linear')(x)
    return ListedColormap(vals)