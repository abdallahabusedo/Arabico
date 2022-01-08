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
import os.path as path
import csv


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def readImageGray(imgPath):
    img = imread(imgPath, as_gray=True)
    img = img.astype(np.float64) / np.max(img)
    img = 255 * img
    img = img.astype(np.uint8)
    return img


def getImgsPaths(_path):
    imgs = [path.join(_path, f) for f in os.listdir(
        _path) if path.isfile(path.join(_path, f))]
    return imgs


def getDataAboutFeatures(imgs, extractFeature):
    extractedFeatures = [extractFeature(img) for img in imgs]
    mean = np.mean(extractedFeatures, axis=0)
    var = np.var(extractedFeatures, axis=0)
    return mean, var


def isTextBlack(_img):
    img = np.copy(_img)
    t = filters.threshold_otsu(img)
    img = img > t
    white_vert = np.sum(img, axis=0)
    white_horz = np.sum(img, axis=1)
    black_vert = img.shape[0]-white_vert
    black_horz = img.shape[1] - white_horz
    white_sum = white_horz[0] + white_horz[-1] + white_vert[0] + white_vert[-1]
    black_sum = black_horz[0] + black_horz[-1] + black_vert[0] + black_vert[-1]
    if black_sum > white_sum:
        return False
    return True


def saveArrayToCSV(arr, filename, label="", append=False):
    file = None
    if append:
        file = open(filename, 'a')
    else:
        file = open(filename, "w")
    row = ""

    for i in range(len(arr)):
        row += str(arr[i])+","
    row += label + "\n"

    file.write(row)


def readFromCSV(filename):
    features = []
    labels = []
    csv_reader = csv.reader(open(filename), delimiter=',')
    for row in csv_reader:
        feature = row[0:-1]
        for i in range(len(feature)):
            feature[i] = float(feature[i])
        feature = np.asarray(feature)
        label = row[-1]
        features.append(feature)
        labels.append(label)

    return np.asarray(features), labels


def euc_dist(X):
    Y = X = X.astype(np.float)
    XX = np.sum(X * X, axis=1)[:, np.newaxis]
    YY = XX.T
    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, distances)
    distances.flat[::distances.shape[0] + 1] = 0.0
    return np.sqrt(distances)
