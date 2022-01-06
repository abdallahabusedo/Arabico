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
