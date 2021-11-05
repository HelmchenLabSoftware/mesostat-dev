import numpy as np
import matplotlib.pyplot as plt
from mesostat.visualization.mpl_colors import base_colors_rgb


# Convert degrees to radians
def _deg2rad(phi):
    return phi * np.pi / 180


# Rotate a vector clockwise around the origin
def _rot(p, phi):
    s = np.sin(phi)
    c = np.cos(phi)
    R = np.array([[c, -s], [s, c]])
    return R.dot(p)


# Get coordinates of equilateral triangle of given origin, radius and rotation
def _uni_triangle_points(p0, rad, phi):
    pRef = np.array([rad, 0])
    p1 = p0 + _rot(pRef, _deg2rad(phi))
    p2 = p0 + _rot(pRef, _deg2rad(phi + 120))
    p3 = p0 + _rot(pRef, _deg2rad(phi + 240))
    return p1, p2, p3


# Construct a line segment between points p1 and p2
# Shift that line segment by sh
# Direction of shift determined by angle phi relative to the original direction of the vector p2-p1
# Return list of x and y coordinates separately
def _sh_line_points(p1, p2, phi, sh):
    v = p2 - p1
    vSh = v / np.linalg.norm(v) * sh   # Normalize
    vShRot = _rot(vSh, phi)            # Rotate
    p1sh = p1 + vShRot                 # Shift
    p2sh = p2 + vShRot                 # Shift
    return [p1sh[0], p2sh[0]], [p1sh[1], p2sh[1]]


def sketch_pid(ax, pidDict, colorsDict=None,
               radiusMacro=3, radiusCircle=1, colorCircle='lightgray', maxLineWidth=15,
               rotation=90, fontsize=30):
    '''
    :param ax:              Plot axis
    :param u1:              Unique information, source X. Allowed values between [0, 1], please rescale
    :param u2:              Unique information, source Y. Allowed values between [0, 1], please rescale
    :param red:             Redundant information, target Z. Allowed values between [0, 1], please rescale
    :param syn:             Synergistic information, target Z. Allowed values between [0, 1], please rescale
    :param radiusMacro:     Radius on which the three circles are placed
    :param radiusCircle:    Radius of each circle
    :param colorCircle:     Color of each circle
    :param maxLineWidth:    Maximum line width for unique and synergistic lines
    :param colorU1:         Color of Unique information, source X
    :param colorU2:         Color of Unique information, source Y
    :param colorRed:        Color of Redundant information, target X
    :param colorSyn:        Color of Synergistic information, target X
    :param rotation:        Rotation of the plot (direction where target is pointing)
    :param fontsize:        Font size for source and target labels
    :return:
    '''


    if colorsDict is None:
        tableauColors = base_colors_rgb(key='tableau')
        colorsDict = {
            'unq_s1'    : tableauColors[0],
            'unq_s2'    : tableauColors[1],
            'shd_s1_s2' : tableauColors[2],
            'syn_s1_s2' : tableauColors[3]
        }

    # Center plot at origin
    p0 = np.array([0, 0])

    ##################################
    # Construct and annotate circle
    ##################################
    pZ, pX, pY = _uni_triangle_points(p0, radiusMacro, rotation)

    circleX = plt.Circle(pX, radius=radiusCircle, color=colorCircle, zorder=2)
    circleY = plt.Circle(pY, radius=radiusCircle, color=colorCircle, zorder=2)
    circleZ = plt.Circle(pZ, radius=radiusCircle, color=colorCircle, zorder=2)

    ax.add_patch(circleX)
    ax.add_patch(circleY)
    ax.add_patch(circleZ)

    labelX = ax.annotate("X", xy=pX, fontsize=fontsize, ha="center", va="center")
    labelY = ax.annotate("Y", xy=pY, fontsize=fontsize, ha="center", va="center")
    labelZ = ax.annotate("Z", xy=pZ, fontsize=fontsize, ha="center", va="center")


    ##################################
    # Construct and annotate Unique and Redundant
    ##################################

    linewidthU1  = maxLineWidth * pidDict['unq_s1']
    linewidthU2  = maxLineWidth * pidDict['unq_s2']
    linewidthRed = maxLineWidth * pidDict['shd_s1_s2']

    lpUnqXZ = _sh_line_points(pX, pZ, _deg2rad(90), radiusCircle / 2)
    lpUnqYZ = _sh_line_points(pY, pZ, _deg2rad(-90), radiusCircle / 2)
    lpRedXZ = _sh_line_points(pX, pZ, _deg2rad(-90), 0)
    lpRedYZ = _sh_line_points(pY, pZ, _deg2rad(90), 0)

    lineUnqXZ = plt.Line2D(*lpUnqXZ, color=colorsDict['unq_s1'], linewidth=linewidthU1, zorder=1)
    lineUnqYZ = plt.Line2D(*lpUnqYZ, color=colorsDict['unq_s2'], linewidth=linewidthU2, zorder=1)
    lineRedXZ = plt.Line2D(*lpRedXZ, color=colorsDict['shd_s1_s2'], linewidth=linewidthRed, zorder=1)
    lineRedYZ = plt.Line2D(*lpRedYZ, color=colorsDict['shd_s1_s2'], linewidth=linewidthRed, zorder=1)

    ax.add_line(lineUnqXZ)
    ax.add_line(lineUnqYZ)
    ax.add_line(lineRedXZ)
    ax.add_line(lineRedYZ)


    ##################################
    # Construct and annotate Synergy
    ##################################

    radiusSynergy = (radiusMacro - radiusCircle) * pidDict['syn_s1_s2']
    pZsyn, pXsyn, pYsyn = _uni_triangle_points(p0, radiusSynergy, rotation)

    triangleSyn = plt.Polygon(np.array([pXsyn, pYsyn, pZsyn]), color=colorsDict['syn_s1_s2'])
    ax.add_patch(triangleSyn)


    ##################################
    # Tuning
    ##################################

    ax.axis('off')
    ax.set_aspect('equal')
    ax.autoscale_view()
