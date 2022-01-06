from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage.morphology import binary_dilation, binary_erosion
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import filters
import math
import os
import helpers
import preprocessing


def get_ToS(skeleton_img, type):
    result = np.copy(skeleton_img)
    if type == "sob":
        result = sk.filters.sobel(skeleton_img)
    elif type == "prew":
        result = sk.filters.prewitt(skeleton_img)
    elif type == "rob":
        result = sk.filters.roberts(skeleton_img)
    elif type == "sch":
        result = sk.filters.scharr(skeleton_img)
    elif type == "lap":
        result = sk.filters.laplace(skeleton_img)
    elif type == "hog":
        _, result = sk.feature.hog(skeleton_img, visualize=True)
    return result


def getGradients(_img, no_of_features=5):
    # print(_img.shape)
    img = np.copy(_img)
    # hog = cv2.HOGDescriptor()
    # c_img = np.uint8(img)
    # h = hog.compute(c_img)
    # h = np.reshape(h, (int(h.shape[0]/9), 9))
    fd = sk.feature.hog(img, orientations=9, pixels_per_cell=(
        16, 16), cells_per_block=(3, 3), transform_sqrt=True, feature_vector=True)
    fd = np.reshape(fd, (int(fd.shape[0]/9), 9))
    mean = np.mean(fd, axis=0)
    return mean
