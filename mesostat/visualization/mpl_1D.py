import numpy as np
import matplotlib.pyplot as plt

from mesostat.visualization.mpl_colors import sample_cmap
from mesostat.visualization.mpl_colorbar import imshow_add_fake_color_bar


# Plot y(x) with color given by z(x)
def plot_coloured_1D(ax, x, y, z, cmap='jet', vmin=None, vmax=None, haveColorBar=False):
    colors = sample_cmap(cmap, z, vmin=vmin, vmax=vmax)
    for i in range(1, len(x)):
        ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=colors[i])

    if haveColorBar:
        imshow_add_fake_color_bar(ax.figure, ax, cmap=cmap, vmin=vmin, vmax=vmax)