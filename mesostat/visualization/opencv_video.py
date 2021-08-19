import os
import numpy as np
import cv2
import matplotlib.image as mpimg


def _append_ext(pathName, ext):
    oldExt = os.path.splitext(pathName)[1]
    if oldExt == ext:
        return pathName
    elif oldExt == "":
        return pathName + ext
    else:
        raise ValueError('Attempting to set old extension', oldExt, 'to', ext)


def _img_matplotlib_2_cv(img3D, isColor=False):
    # Make sure image is UINT8 and spans values [0, 255]
    img = np.uint8(img3D) if np.max(img3D) > 1 else np.uint8(255 * img3D)

    # Select correct color ordering, or only one color if grayscale
    colorReorder = np.array([2, 1, 0])  # OpenCV and Matplotlib seem to disagree about color order in RGB
    img = img[:, :, colorReorder] if isColor else img[:, :, 0]

    return img


# Convert a set of images to a video
def merge_images_cv2(srcPaths, trgPathName, fps=30, FOURCC='MJPG', isColor=False):
    print("Writing video to", trgPathName)

    if FOURCC not in ['MJPG', 'XVID']:
        raise ValueError("Unexpected target encoding", FOURCC)

    # Load 1 picture to get its shape
    img = mpimg.imread(srcPaths[0])

    # Convert between standards of different libraries
    shape2Dcv = (img.shape[1], img.shape[0])   # OpenCV uses column-major or sth

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    outPathNameEff = _append_ext(trgPathName, '.avi')
    out = cv2.VideoWriter(outPathNameEff, fourcc, fps, shape2Dcv, isColor=isColor)

    for iSrc, srcPath in enumerate(srcPaths):
        print('Processing image[%d]\r' % iSrc, end="")
        imgSrc = mpimg.imread(srcPath)
        imgSrc = _img_matplotlib_2_cv(imgSrc, isColor=isColor)

        out.write(imgSrc)

    print("\n Done")

    out.release()
