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
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


def decisionTreeClassifier(features, labels):
    # load features from csv files
    # labels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    return clf


def svmClassifier(features, labels):
    # clf = svm.NuSVC(gamma="auto")
    parameters = {'kernel': (
        "linear", "poly", "rbf", "sigmoid"), 'C': [1, 10,100], 'degree': [3,6,9,20]}
    svc = svm.SVC(cache_size=500)
    clf = make_pipeline(StandardScaler(), GridSearchCV(svc, parameters))
    clf.fit(features, labels)
    return clf


def svmNuClassifier(features, labels):
    # clf = svm.NuSVC(gamma="auto")
    parameters = {'kernel': (
        "linear", "poly", "rbf", "sigmoid"), 'nu': [0.4, 0.5, 0.6]}
    nusvc = svm.NuSVC()
    clf = make_pipeline(StandardScaler(), GridSearchCV(nusvc, parameters))
    clf.fit(features, labels)
    return clf


def adaboostClassifier(features, labels):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(features, labels)
    return clf


def randomForestClassifier(features, labels):
    parameters = {'n_estimators': [10,100,1000], 'max_features': ["auto", "sqrt", "log2"]}
    mlp = RandomForestClassifier()
    clf = make_pipeline(StandardScaler(), GridSearchCV(mlp, parameters))
    clf.fit(features, labels)
    return clf


def NNClassifier(features, labels):
    parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                  'learning_rate': ['constant','invscaling','adaptive'],
                  'max_iter': [1000,1500],
                  'validation_fraction':[0.1,0.2,0.3]
                  }
    mlp = MLPClassifier(solver='lbfgs',alpha=1e-5, random_state=1, hidden_layer_sizes=(9,))
    clf = make_pipeline(StandardScaler(), GridSearchCV(mlp, parameters))
    clf.fit(features, labels)
    return clf
# def NNClassifier(features, labels):
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                         hidden_layer_sizes=(9,), random_state=1,max_iter=1000)
#     clf.fit(features, labels)
#     return clf
