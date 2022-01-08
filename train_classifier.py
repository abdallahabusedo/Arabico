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
# hvsl, _ = helpers.readFromCSV("csv/hvsl_new_1.csv")

lvl_test, _ = helpers.readFromCSV("csv/lvl_test.csv")
hpp_test, labels_test = helpers.readFromCSV("csv/hpp_test.csv")
lpq_test, _ = helpers.readFromCSV("csv/lpq_test.csv")
toe_test, _ = helpers.readFromCSV("csv/toe_test.csv")
tos_test, _ = helpers.readFromCSV("csv/tos_test.csv")
wor_test, _ = helpers.readFromCSV("csv/wor_test.csv")
hvsl_test, _ = helpers.readFromCSV("csv/hvsl_test.csv")
# hvsl_test, _ = helpers.readFromCSV("csv/hvsl_test_new_1_1.csv")

lvl_test_new, _ = helpers.readFromCSV("csv/lvl_test_new.csv")
hpp_test_new, labels_test_new = helpers.readFromCSV("csv/hpp_test_new.csv")
lpq_test_new, _ = helpers.readFromCSV("csv/lpq_test_new.csv")
toe_test_new, _ = helpers.readFromCSV("csv/toe_test_new.csv")
tos_test_new, _ = helpers.readFromCSV("csv/tos_test_new.csv")
wor_test_new, _ = helpers.readFromCSV("csv/wor_test_new.csv")
hvsl_test_new, _ = helpers.readFromCSV("csv/hvsl_test_new.csv")
# hvsl_test_new, _ = helpers.readFromCSV("csv/hvsl_test_new_1.csv")

# print(len(toe[0]), len(tos[0]), len(hpp[0]),
#       len(lvl[0]), len(hvsl[0]), len(wor[0]))
allFeatures = np.concatenate(tuple([toe, tos, wor,	hvsl, hpp, lvl]), axis=1)
allFeaturesTest = np.concatenate(tuple([toe_test, tos_test, wor_test,	hvsl_test, hpp_test, lvl_test]), axis=1)
allFeaturesTestNew = np.concatenate(tuple([toe_test_new, tos_test_new, wor_test_new,	hvsl_test_new, hpp_test_new, lvl_test_new]), axis=1)

clf = classification.svmClassifier(allFeatures, labels)
# clf = classification.NNClassifier(allFeatures, labels)
# clf = classification.randomForestClassifier(allFeatures, labels)

print("training=",clf.score(allFeatures,labels))
print("test=",clf.score(allFeaturesTest,labels_test))
print("test new=",clf.score(allFeaturesTestNew,labels_test_new))
helpers.saveClfParameters(clf)
