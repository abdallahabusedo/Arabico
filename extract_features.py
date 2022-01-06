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


def getGradients(_img_bin, no_of_features=5):
    # print(_img.shape)
    try:
        img = np.copy(_img_bin)
        sy = filters.sobel_h(img)+0.0000000000000000000000000000000000001
        sx = filters.sobel_v(img)+0.0000000000000000000000000000000000001
        z = np.rad2deg(np.arctan(sy/sx))
        z = z.astype(int)
        # -180 -135 -90 -45 0 45 90 135 180
        n_bins = 9
        #delta_angle = 180//n_bins
        hist = np.zeros(n_bins)
        for k in range(n_bins):
            upper_bound = (k+1)*45 - 180
            lower_bound = k*45 - 180
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if z[i][j] < upper_bound and z[i][j] >= lower_bound:
                        hist[k] += 1

        hist /= (img.shape[0]*img.shape[1])
        return hist
    except Exception as ex:
        print("error image")
        print(img.shape)
        print(ex)
        return np.zeros(9)


def getHPP(_img_gray):
    img = np.copy(_img_gray)
    n_bins = 10
    height = 50
    img = resize(img, (height, height),
                 anti_aliasing=True)
    img = preprocessing.binarization(img)
    img = img == False
    horz_sum = np.sum(img, axis=1)
    f = np.zeros(n_bins)
    for i in range(n_bins):
        for j in range(int(height//n_bins)):
            f[i] += horz_sum[i*int(height//n_bins)+j]
    f /= (height*height)
    return f
