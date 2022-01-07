from numpy.core.fromnumeric import shape
import helpers
import classification
import numpy as np
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


lvl, labels = helpers.readFromCSV("csv/lvl.csv")  # LVL is not usable
hpp, _ = helpers.readFromCSV("csv/hpp.csv")
lpq, _ = helpers.readFromCSV("csv/lpq.csv")
toe, _ = helpers.readFromCSV("csv/toe.csv")
tos, _ = helpers.readFromCSV("csv/tos.csv")
#tth, _ = helpers.readFromCSV("tth.csv")
wor, _ = helpers.readFromCSV("csv/wor.csv")

lvl_test, labels_test = helpers.readFromCSV(
    "csv/lvl_test.csv")  # LVL is not usable
hpp_test, _ = helpers.readFromCSV("csv/hpp_test.csv")
lpq_test, _ = helpers.readFromCSV("csv/lpq_test.csv")
toe_test, _ = helpers.readFromCSV("csv/toe_test.csv")
tos_test, _ = helpers.readFromCSV("csv/tos_test.csv")
#tth, _ = helpers.readFromCSV("tth.csv")
wor_test, _ = helpers.readFromCSV("csv/wor_test.csv")


features = (lpq, wor, hpp, toe, tos)
allFeatures = np.concatenate(
    features, axis=1)

features_test = (lpq_test, wor_test, hpp_test, toe_test, tos_test)
allFeatures_test = np.concatenate(
    features_test, axis=1)

clf = classification.svmClassifier(allFeatures, labels)
acc = 0
for i in range(allFeatures.shape[0]):
    f = np.reshape(allFeatures[i], (1, -1))
    if clf.predict(f) == labels[i]:
        acc += 1
print("test accuracy ", 100*acc/len(labels))


acc = 0
for i in range(allFeatures_test.shape[0]):
    f = np.reshape(allFeatures_test[i], (1, -1))
    if clf.predict(f) == labels_test[i]:
        acc += 1
print("test accuracy ", 100*acc/len(labels_test))
