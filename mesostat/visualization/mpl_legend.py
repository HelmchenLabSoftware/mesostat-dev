import matplotlib.patches as mpatches


def plt_add_fake_legend(ax, colors, labels, loc=None, bbox_to_anchor=None):
    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, labels)]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor)
