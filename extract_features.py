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


def get_LVL(skeleton_img, lineThresholdFraction=0.2):
    result = np.zeros(skeleton_img.shape)
    rows, cols = result.shape
    featureVector = []
    verticalLines = []

    for col in range(cols):
        cnt = 0
        tempCnt = 0
        l, r = -1, -1
        tl, tr = -1, -1
        minl, maxr = 1e9, 0

        for row in range(rows):
            if skeleton_img[row, col] == 1:
                tempCnt += 1
                minl = min(minl, row)
                maxr = max(maxr, row)
                if tl == -1:
                    tl = row
                else:
                    tr = row
            else:
                if tempCnt > cnt:
                    l, r = tl, tr
                    cnt = tempCnt
                tempCnt = 0
        if cnt > rows*lineThresholdFraction:
            verticalLines.append(cnt)
            result[l:r+1, col] = 1

    if len(verticalLines) == 0:
        verticalLines.append(0)
    verticalLines = np.array(verticalLines)

    featureVector.append(maxr-minl+1)  # text height
    featureVector.append(verticalLines.shape[0])  # number of lines
    featureVector.append(np.max(verticalLines))  # longest line
    featureVector.append(np.max(verticalLines)/(maxr-minl+1))  # ratio between longest line and text height
    featureVector.append(np.var(verticalLines))  # variance among lines
    return featureVector, result


def get_TTH(skeleton_img, minThicknessThreshold=5, maxThicknessThreshold=100):
    result = np.zeros(skeleton_img.shape)
    rows, cols = skeleton_img.shape
    allThickness = []
    for row in range(rows):
        tl, tr = -1, -1
        for col in range(cols):
            if skeleton_img[row, col] == 1:
                if tl == -1:
                    tl = col
                else:
                    tr = col
            else:
                if tr-tl+1 > minThicknessThreshold and tr-tl+1 < maxThicknessThreshold:
                    allThickness.append(tr-tl+1)
                    result[row, tl:tr+1] = 1
                tl, tr = -1, -1
    print((allThickness))
    return allThickness, result


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
