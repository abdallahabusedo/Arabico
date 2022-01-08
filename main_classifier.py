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
hvsl, _ = helpers.readFromCSV("csv/hvsl.csv")

lvl_test, labels_test = helpers.readFromCSV(
    "csv/lvl_test.csv")  # LVL is not usable
hpp_test, _ = helpers.readFromCSV("csv/hpp_test.csv")
lpq_test, _ = helpers.readFromCSV("csv/lpq_test.csv")
toe_test, _ = helpers.readFromCSV("csv/toe_test.csv")
tos_test, _ = helpers.readFromCSV("csv/tos_test.csv")
#tth, _ = helpers.readFromCSV("tth.csv")
wor_test, _ = helpers.readFromCSV("csv/wor_test.csv")
hvsl_test, _ = helpers.readFromCSV("csv/hvsl_test.csv")

features_name = ['lpq', 'toe', 'tos', 'wor', 'hslv', 'hpp']
feature_values = [lpq, toe, tos, wor, hvsl, hpp]
feature_values_test = [lpq_test, toe_test, tos_test,
                       wor_test, hvsl_test, hpp_test]
highest_acc = -1
f_used = None
for i in range(1, int(2**len(features_name))+1):
    features_used = []
    features_used_test = []
    features_used_indx = []
    bin_string = bin(i)[::-1]
    for i in range(len(bin_string)):
        if bin_string[i] == 'b':
            break
        if bin_string[i] == '1':
            indx = len(features_name)-i-1
            features_used.append(feature_values[indx])
            features_used_test.append(feature_values_test[indx])
            features_used_indx.append(indx)
    print("features used")
    for indx in features_used_indx:
        print(features_name[indx])
    if len(features_used) == 0:
        continue
    elif len(features_used) == 1:
        clf = classification.svmClassifier(features_used[0], labels)
        #clf = classification.adaboostClassifier(features_used[0], labels)
        #clf = classification.randomForestClassifier(features_used[0], labels)
        # get score
        score = clf.score(features_used[0], labels)
        print("train score", score)
        score = clf.score(features_used_test[0], labels_test)
        print("test score", score)
        acc = 0
        for i in range(features_used[0].shape[0]):
            f = np.reshape(features_used[0][i], (1, -1))
            if clf.predict(f) == labels[i]:
                acc += 1
        print("train accuracy ", 100*acc/len(labels))
        if 100*acc/len(labels) > highest_acc:
            highest_acc = 100*acc/len(labels)
            f_used = features_used_indx
        acc = 0
        for i in range(features_used_test[0].shape[0]):
            f = np.reshape(features_used_test[0][i], (1, -1))
            if clf.predict(f) == labels_test[i]:
                acc += 1
        print("test accuracy ", 100*acc/len(labels_test))

    else:
        allFeatures = np.concatenate(tuple(features_used), axis=1)
        allFeatures_test = np.concatenate(
            tuple(features_used_test), axis=1)

        clf = classification.svmClassifier(allFeatures, labels)
        #clf = classification.adaboostClassifier(allFeatures, labels)
        #clf = classification.randomForestClassifier(allFeatures, labels)

        score = clf.score(allFeatures, labels)
        print("train score", score)
        score = clf.score(allFeatures_test, labels_test)
        print("test score", score)

        acc = 0
        for i in range(allFeatures.shape[0]):
            f = np.reshape(allFeatures[i], (1, -1))
            if clf.predict(f) == labels[i]:
                acc += 1
        print("train accuracy ", 100*acc/len(labels))
        if 100*acc/len(labels) > highest_acc:
            highest_acc = 100*acc/len(labels)
            f_used = features_used_indx
        acc = 0
        for i in range(allFeatures_test.shape[0]):
            f = np.reshape(allFeatures_test[i], (1, -1))
            if clf.predict(f) == labels_test[i]:
                acc += 1
        print("test accuracy ", 100*acc/len(labels_test))

print("\n\n\nhighest training acc  ", highest_acc)
print("when using")
for indx in f_used:
    print(features_name[indx])
