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


lvl, labels = helpers.readFromCSV("lvl.csv")  # LVL is not usable
hpp, _ = helpers.readFromCSV("hpp.csv")
lpq, _ = helpers.readFromCSV("lpq.csv")
toe, _ = helpers.readFromCSV("toe.csv")
tos, _ = helpers.readFromCSV("tos.csv")
#tth, _ = helpers.readFromCSV("tth.csv")
wor, _ = helpers.readFromCSV("wor.csv")

lvl_test, labels = helpers.readFromCSV("lvl_test.csv")  # LVL is not usable
hpp_test, _ = helpers.readFromCSV("hpp_test.csv")
lpq_test, _ = helpers.readFromCSV("lpq_test.csv")
toe_test, _ = helpers.readFromCSV("toe_test.csv")
tos_test, _ = helpers.readFromCSV("tos_test.csv")
#tth, _ = helpers.readFromCSV("tth.csv")
wor_test, _ = helpers.readFromCSV("wor_test.csv")


# features = (lpq, wor, hpp, toe, tos)
# allFeatures = np.concatenate(
#     features, axis=1)
# print(allFeatures.shape)
# clf = classification.svmClassifier(allFeatures, labels)
# acc = 0
# for i in range(allFeatures.shape[0]):
#     f = np.reshape(allFeatures[i], (1, -1))
#     if clf.predict(f) == labels[i]:
#         acc += 1
# print(100*acc/len(labels))
# for x in tth:
#     print(x.shape)
# print(lvl.shape, hpp.shape, lpq.shape,
#       toe.shape, tos.shape, tth.shape, wor.shape)
# np.concatenate((lvl, hpp), axis=1)
