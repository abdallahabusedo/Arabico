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
    try:

        img = np.copy(_img)
        # hog = cv2.HOGDescriptor()
        # c_img = np.uint8(img)
        # h = hog.compute(c_img)
        # h = np.reshape(h, (int(h.shape[0]/9), 9))
        # fd = sk.feature.hog(img, orientations=9, pixels_per_cell=(
        #     16, 16), cells_per_block=(3, 3), transform_sqrt=True, feature_vector=True)
        # fd = np.reshape(fd, (int(fd.shape[0]/9), 9))
        # mean = np.mean(fd, axis=0)
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
            #print(upper_bound, lower_bound)
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
