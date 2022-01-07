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
appendCSV = False
for i in range(1, 10):
    print("\ndataset "+str(i))
    imgsPaths = helpers.getImgsPaths("./ACDB/ACdata_base/"+str(i))
    imgs = []
    # i = 0
    for imgPath in imgsPaths:
        img = helpers.readImageGray(imgPath)
        isTextBlack = helpers.isTextBlack(img)
        arr = extract_features.getLPQ(img)
        # helpers.saveArrayToCSV(arr, "lpq.csv", str(i), appendCSV)
        # arr = extract_features.getHPP(img)
        # helpers.saveArrayToCSV(arr, "hpp.csv", str(i), appendCSV)
        img_bin = preprocessing.binarization(img, isTextBlack)
        img_skeleton = preprocessing.skeletonization(img_bin)
        # arr = extract_features.getGradients(img_skeleton)
        # helpers.saveArrayToCSV(arr, "tos.csv", str(i), appendCSV)
        # _, Bimg = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        # img_edge = preprocessing.getEdgeImage(Bimg)
        # arr = extract_features.getGradients(img_edge)
        # helpers.saveArrayToCSV(arr, "toe.csv", str(i), appendCSV)
        # arr = extract_features.getWOr(img)
        # helpers.saveArrayToCSV(arr, "wor.csv", str(i), appendCSV)
        arr = extract_features.getLVL(img_skeleton)
        helpers.saveArrayToCSV(arr, "csv/lvl.csv", str(i), appendCSV)
        # arr = extract_features.getTTH(img_skeleton)
        # helpers.saveArrayToCSV(arr, "tth.csv", str(i), appendCSV)
        # arr = extract_features.getHVSL(img, isTextBlack)
        # helpers.saveArrayToCSV(arr, "csv/hvsl.csv", str(i), appendCSV)
        appendCSV = True
        # helpers.show_images([img])
        # imgs.append(img)
        # break

        # mean, var = helpers.getDataAboutFeatures(
        #     [img], extract_features.getGradients)
        # print("mean\t", "var")
        # for x in range(len(mean)):
        #     print(mean[x], "\t", var[x])
        # if i == 30:
        #     break
        # i += 1
    # print(imgs[0].shape)
    # mean, var = helpers.getDataAboutFeatures(
    #     imgs, extract_features.getLPQ)
    # print("mean\t\t\t\t", "var")
    # for i in range(len(mean)):
    #     print(mean[i], "\t\t\t\t", var[i])
# get test data
appendCSV = False
for i in range(1, 10):
    print("\ndataset "+str(i))
    imgsPaths = helpers.getImgsPaths("./test_images/"+str(i))
    # i = 0
    for imgPath in imgsPaths:
        img = helpers.readImageGray(imgPath)
        isTextBlack = helpers.isTextBlack(img)
        # arr = extract_features.getLPQ(img)
        # helpers.saveArrayToCSV(arr, "lpq_test.csv", str(i), appendCSV)
        # arr = extract_features.getHPP(img)
        # helpers.saveArrayToCSV(arr, "hpp_test.csv", str(i), appendCSV)
        # img_bin = preprocessing.binarization(img, isTextBlack)
        # img_skeleton = preprocessing.skeletonization(img_bin)
        # arr = extract_features.getGradients(img_skeleton)
        # helpers.saveArrayToCSV(arr, "tos_test.csv", str(i), appendCSV)
        # _, Bimg = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        # img_edge = preprocessing.getEdgeImage(Bimg)
        # arr = extract_features.getGradients(img_edge)
        # helpers.saveArrayToCSV(arr, "toe_test.csv", str(i), appendCSV)
        # arr = extract_features.getWOr(img)
        # helpers.saveArrayToCSV(arr, "wor_test.csv", str(i), appendCSV)
        # arr = extract_features.getLVL(img_skeleton)
        # helpers.saveArrayToCSV(arr, "lvl_test.csv", str(i), appendCSV)
        # arr = extract_features.getTTH(img_skeleton)
        # helpers.saveArrayToCSV(arr, "tth_test.csv", str(i), appendCSV)
        arr = extract_features.getHVSL(img, isTextBlack)
        helpers.saveArrayToCSV(arr, "csv/hvsl_test.csv", str(i), appendCSV)
        appendCSV = True
