from numpy.core.fromnumeric import shape
import helpers
import classification
import numpy as np
lvl, labels = helpers.readFromCSV("lvl.csv")  # LVL is not usable
hpp, _ = helpers.readFromCSV("hpp.csv")
lpq, _ = helpers.readFromCSV("lpq.csv")
toe, _ = helpers.readFromCSV("toe.csv")
tos, _ = helpers.readFromCSV("tos.csv")
#tth, _ = helpers.readFromCSV("tth.csv")
wor, _ = helpers.readFromCSV("wor.csv")
features = (lpq, wor, hpp, toe, tos)
allFeatures = np.concatenate(
    features, axis=1)
print(allFeatures.shape)
clf = classification.svmClassifier(allFeatures, labels)
acc = 0
for i in range(allFeatures.shape[0]):
    f = np.reshape(allFeatures[i], (1, -1))
    if clf.predict(f) == labels[i]:
        acc += 1
print(100*acc/len(labels))
# for x in tth:
#     print(x.shape)
# print(lvl.shape, hpp.shape, lpq.shape,
#       toe.shape, tos.shape, tth.shape, wor.shape)
# np.concatenate((lvl, hpp), axis=1)
