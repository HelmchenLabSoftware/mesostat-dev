import numpy as np
import matplotlib.pyplot as plt

from mesostat.visualization.mpl_colorbar import imshow_add_fake_color_bar


# Plot y(x) with color given by z(x)
def plot_coloured_1D(ax, x, y, z, cmap='jet', vmin=None, vmax=None, haveColorBar=False):
    nP = len(x)

    vminEff = np.min(z) if vmin is None else vmin
    vmaxEff = np.max(z) if vmax is None else vmax

    zNorm = (z - vminEff) / (vmaxEff - vminEff)
    zNorm = np.clip(zNorm, 0, 1)

    cmapFunc = plt.get_cmap(cmap)
    for i in range(1, nP):
        # ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=(zNorm[i], 0, 0))
        ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=cmapFunc(zNorm[i]))

    if haveColorBar:
        imshow_add_fake_color_bar(ax.figure, ax, cmap=cmap, vmin=vmin, vmax=vmax)