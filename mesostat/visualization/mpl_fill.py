from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


# Fill between two values of x, full height of the plot
def fill_between_x(ax, xStartLst, xEndLst, color, alpha=1.0):
    y0, y1, = ax.get_ylim()
    patches = [Rectangle((x0, y0), x1-x0, y1-y0) for x0, x1 in zip(xStartLst, xEndLst)]
    ax.add_collection(PatchCollection(patches, color=color, alpha=alpha, edgecolor='none'))


# Fill between two values of y, full length of the plot
def fill_between_y(ax, yStartLst, yEndLst, color, alpha=1.0):
    x0, x1, = ax.get_xlim()
    patches = [Rectangle((x0, y0), x1-x0, y1-y0) for y0, y1 in zip(yStartLst, yEndLst)]
    ax.add_collection(PatchCollection(patches, color=color, alpha=alpha, edgecolor='none'))
