from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
#import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage.morphology import binary_dilation, binary_erosion
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import filters
import math
import os
import cv2


def binarization(_img):
    img = np.copy(_img)
    t = sk.filters.threshold_local(img, 21, offset=10)
    img = img < t
    return img


def get_skeleton(_input_img, reverse_bin=False):
    input_img = np.copy(_input_img)
    # convert to grey scale
    input_img = input_img.astype(np.float64) / np.max(input_img)
    input_img = 255 * input_img
    input_img = input_img.astype(np.uint8)

    # binarization
    input_img = binarization(input_img)

    # remove noise
    input_img = binary_dilation(input_img)
    input_img = binary_erosion(input_img)

    # calculate the skeleton
    skeleton_img = sk.morphology.skeletonize(input_img)
    skeleton_img = skeleton_img*1
    if reverse_bin:
        skeleton_img = skeleton_img == 0
        skeleton_img = skeleton_img*1
    return skeleton_img


def skeletonization(_img):
    img = np.copy(_img)
    thin = sk.morphology.skeletonize(img)
    thin = thin*1
    thin = thin == 0
    thin = thin*1
    img = img == 0
    return thin


def getEdgeImg(_img):
    img = np.copy(_img)
    # _,bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    horizontal = np.copy(edges)
    vertical = np.copy(edges)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # print(horizontal_size)
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    return horizontal, vertical
