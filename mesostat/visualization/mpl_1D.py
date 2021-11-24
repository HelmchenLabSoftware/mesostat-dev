import numpy as np
import matplotlib.pyplot as plt

from mesostat.visualization.mpl_colors import sample_cmap
from mesostat.visualization.mpl_colorbar import imshow_add_fake_color_bar


def prettify_plot_1D(ax, haveTopRightBox=True, margins=None,
                     xTicks=None, yTicks=None,
                     xRotation=None, yRotation=None,
                     xFontSize=None, yFontSize=None):
    if not haveTopRightBox:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if margins is not None:
        ax.margins(margins)

    # Font size
    if xFontSize is not None:
        ax.tick_params(axis='x', labelsize=xFontSize)
    if yFontSize is not None:
        ax.tick_params(axis='y', labelsize=yFontSize)

    if xTicks is not None:
        ax.set_xticks(xTicks)
    if yTicks is not None:
        ax.set_yticks(yTicks)

    if xRotation is not None:
        plt.setp(ax.get_xticklabels(), rotation=xRotation, horizontalalignment='center')

    if yRotation is not None:
        plt.setp(ax.get_yticklabels(), rotation=yRotation, horizontalalignment='center')


# Plot y(x) with color given by z(x)
def plot_coloured_1D(ax, x, y, z, cmap='jet', vmin=None, vmax=None, haveColorBar=False):
    colors = sample_cmap(cmap, z, vmin=vmin, vmax=vmax)
    for i in range(1, len(x)):
        ax.plot(x[i - 1:i + 1], y[i - 1:i + 1], color=colors[i])

    if haveColorBar:
        imshow_add_fake_color_bar(ax.figure, ax, cmap=cmap, vmin=vmin, vmax=vmax)