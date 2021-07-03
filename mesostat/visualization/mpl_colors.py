import numpy as np
import colorsys
from scipy import interpolate
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def base_colors_rgb(key='base'):
    if key == 'base':
        colorDict = mcolors.BASE_COLORS
    elif key == 'tableau':
        colorDict = mcolors.TABLEAU_COLORS
    elif key == 'css4':
        colorDict = mcolors.CSS4_COLORS
    elif key == 'xkcd':
        colorDict = mcolors.CSS4_COLORS
    else:
        raise ValueError('Unknown color scheme')

    return [mcolors.to_rgb(v) for c, v in colorDict.items()]


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
    return mcolors.ListedColormap(vals)


def sample_cmap(cmap, arr, vmin=None, vmax=None, dropAlpha=False):
    arrTmp = np.array(arr)

    # Test if samples in correct range
    if vmin is None:
        vmin = np.min(arrTmp)

    if vmax is None:
        vmax = np.max(arrTmp)

    arrNorm = (arrTmp - vmin) / (vmax - vmin)
    arrNorm = np.clip(arrNorm, 0, 1)

    cmapFunc = plt.get_cmap(cmap)
    rez = [cmapFunc(elNorm) for elNorm in arrNorm]
    if not dropAlpha:
        return rez
    else:
        return [r[:3] for r in rez]


def rgb_change_color(img, c1, c2):
    rez = img.copy()
    r,g,b = img.T
    white_areas = (r == c1[0]) & (b == c1[1]) & (g == c1[2])
    rez[white_areas.T] = c2
    return rez