from matplotlib import colors, colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Add colorbar to existing imshow
def imshow_add_color_bar(fig, ax, img):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')


# Adds fake colorbar to any axis. That colorbar will linearly interpolate an existing colormap
def imshow_add_fake_color_bar(fig, ax, cmap, vmin=0, vmax=1):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')