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
import helpers
import preprocessing
import extract_features

for i in range(1, 10):
    print("dataset "+str(i))
    imgsPaths = helpers.getImgsPaths("./ACDB/ACdata_base/"+str(i))
    imgs = []
    i = 0
    for imgPath in imgsPaths:
        img = helpers.readImageGray(imgPath)
        img = preprocessing.binarization(img)
        img = preprocessing.skeletonization(img)
        imgs.append(img)
        # mean, var = helpers.getDataAboutFeatures(
        #     [img], extract_features.getGradients)
        # print("mean\t", "var")
        # for x in range(len(mean)):
        #     print(mean[x], "\t", var[x])
        if i == 30:
            break
        i += 1
    # print(imgs[0].shape)
    mean, var = helpers.getDataAboutFeatures(
        imgs, extract_features.getGradients)
    print("mean\t", "var")
    for i in range(len(mean)):
        print(mean[i], "\t", var[i])