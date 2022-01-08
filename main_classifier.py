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

lvl_test, _ = helpers.readFromCSV("csv/lvl_test.csv")
hpp_test, labels_test = helpers.readFromCSV("csv/hpp_test.csv")
lpq_test, _ = helpers.readFromCSV("csv/lpq_test.csv")
toe_test, _ = helpers.readFromCSV("csv/toe_test.csv")
tos_test, _ = helpers.readFromCSV("csv/tos_test.csv")
wor_test, _ = helpers.readFromCSV("csv/wor_test.csv")
hvsl_test, _ = helpers.readFromCSV("csv/hvsl_test.csv")

features_name = ['lpq', 'toe', 'tos', 'wor', 'hvsl', 'hpp', 'lvl']
feature_values = [lpq, toe, tos, wor, hvsl, hpp, lvl]
feature_values_test = [lpq_test, toe_test, tos_test,
                       wor_test, hvsl_test, hpp_test, lvl_test]
highest_acc = -1
h_a_train = None
f_used = None
# used for comparison CSV file
arr_comp = ['lpq', 'toe', 'tos', 'wor', 'hvsl', 'hpp', 'lvl',
            'accuracy_train', 'accuracy_test', 'accuracy_validation']
helpers.saveArrayToCSV(arr_comp, "results/svm_gridsearch_svc.csv", "", False)
for i in range(1, int(2**len(features_name))+1):
    ###
    arr_comp = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    ###

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
            arr_comp[indx] = 'X'
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
        score_train = clf.score(features_used[0], labels)*100
        print("train score", score_train)
        score_test = clf.score(features_used_test[0], labels_test)*100
        print("test score", score_test)
        if score_test > highest_acc:
            highest_acc = score_test
            f_used = features_used_indx
            h_a_train = score_train
        arr_comp[-3] = score_train
        arr_comp[-2] = score_test

    else:
        allFeatures = np.concatenate(tuple(features_used), axis=1)
        allFeatures_test = np.concatenate(
            tuple(features_used_test), axis=1)

        clf = classification.svmClassifier(allFeatures, labels)
        #clf = classification.adaboostClassifier(allFeatures, labels)
        #clf = classification.randomForestClassifier(allFeatures, labels)

        score_train = clf.score(allFeatures, labels)*100
        print("train score", score_train)
        score_test = clf.score(allFeatures_test, labels_test)*100
        print("test score", score_test)
        if score_test > highest_acc:
            highest_acc = score_test
            f_used = features_used_indx
            h_a_train = score_train
        arr_comp[-3] = score_train
        arr_comp[-2] = score_test
    helpers.saveArrayToCSV(
        arr_comp, "results/svm_gridsearch_svc.csv", "", True)

print("\n\n\nhighest test acc  ", highest_acc)
print("with train acc  ", h_a_train)
print("when using")
for indx in f_used:
    print(features_name[indx])
