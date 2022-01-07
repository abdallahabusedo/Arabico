from numpy.core.fromnumeric import shape
import helpers
import classification
import numpy as np
lvl, labels = helpers.readFromCSV("lvl.csv")
hpp, _ = helpers.readFromCSV("hpp.csv")
lpq, _ = helpers.readFromCSV("lpq.csv")
toe, _ = helpers.readFromCSV("toe.csv")
tos, _ = helpers.readFromCSV("tos.csv")
tth, _ = helpers.readFromCSV("tth.csv")
wor, _ = helpers.readFromCSV("wor.csv")

# allFeatures = np.concatenate(
#     (lvl, hpp, lpq, toe, tos, tth, wor), axis=1)
# clf = classification.svmClassifier(allFeatures, labels)
# print(clf.predict(allFeatures[0]), labels[0])
print(tth[0])
print(lvl.shape, hpp.shape, lpq.shape,
      toe.shape, tos.shape, tth.shape, wor.shape)
# np.concatenate((lvl, hpp), axis=1)
