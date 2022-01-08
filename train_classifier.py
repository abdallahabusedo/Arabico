import helpers
import classification
import numpy as np
import helpers


lvl, _ = helpers.readFromCSV("csv/lvl.csv")
hpp, labels = helpers.readFromCSV("csv/hpp.csv")
lpq, _ = helpers.readFromCSV("csv/lpq.csv")
toe, _ = helpers.readFromCSV("csv/toe.csv")
tos, _ = helpers.readFromCSV("csv/tos.csv")
wor, _ = helpers.readFromCSV("csv/wor.csv")
hvsl, _ = helpers.readFromCSV("csv/hvsl.csv")

# print(len(toe[0]), len(tos[0]), len(hpp[0]),
#       len(lvl[0]), len(hvsl[0]), len(wor[0]))
allFeatures = np.concatenate(tuple([toe, tos, wor,	hvsl, hpp, lvl]), axis=1)
clf = classification.svmClassifier(allFeatures, labels)
helpers.saveClfParameters(clf)
