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
from sklearn import tree


def decisionTreeClassifier():
    # load features from csv files
    wor_features, wor_labels = helpers.readFromCSV("wor.csv")
    # labels = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    wor_features = np.array(wor_features)
    wor_features = np.reshape(wor_features, (-1, 1))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(wor_features, wor_labels)
    # test the classifier and failed X'(
    img = helpers.readImageGray("1496.jpg")
    f = extract_features.getWOr(img)
    f = np.reshape(f, (1, -1))
    # class 6 and the image from class 9
    print(clf.predict(f))