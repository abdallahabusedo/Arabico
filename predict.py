import argparse
import helpers
import preprocessing
import extract_features
import cv2
import numpy as np
import time
import math


def main(in_name, out_name):
    result_file_name = "results.txt"
    time_file_name = "times.txt"
    result_file = open(out_name+result_file_name, "w")
    time_file = open(out_name+time_file_name, "w")
    result_file.write("")
    time_file.write("")
    result_file.close()
    time_file.close()
    result_file = open(out_name+result_file_name, "a")
    time_file = open(out_name+time_file_name, "a")
    clf = helpers.loadCLfParameters("finalized_classifier.sav")
    imgsPaths = helpers.getImgsPaths(in_name)
    for imgPath in imgsPaths:
        st_time = time.time()
        try:
            img = helpers.readImageGray(imgPath)
            isTextBlack = helpers.isTextBlack(img)
            lpq = extract_features.getLPQ(img)
            hpp = extract_features.getHPP(img)
            img_bin = preprocessing.binarization(img, isTextBlack)
            img_skeleton = preprocessing.skeletonization(img_bin)
            tos = extract_features.getGradients(img_skeleton)
            _, Bimg = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
            img_edge = preprocessing.getEdgeImage(Bimg)
            toe = extract_features.getGradients(img_edge)
            wor = extract_features.getWOr(img)
            lvl = extract_features.getLVL(img_skeleton)
            hvsl = extract_features.getHVSL(img, isTextBlack)
            allFeatures = np.concatenate((toe, tos, wor, hvsl, hpp, lvl))

            # print(len(toe), len(tos), len(hpp), len(lvl), len(hvsl), len(wor))
            # print(allFeatures.shape)
            fv = np.reshape(allFeatures, (1, -1))
            label_predicted = clf.predict(fv)
            print(label_predicted[0])
            end_time = time.time()-st_time
            end_time = max(0.001, end_time)
            # save to result time
            result_file.write(label_predicted[0]+"\n")
            time_file.write(str(end_time)+"\n")
        except Exception as ex:
            print("exception", ex)
            end_time = time.time()-st_time
            end_time = max(0.001, end_time)
            result_file.write(str(-1)+"\n")
            time_file.write(str(end_time)+"\n")

    result_file.close()
    time_file.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", help="input filename")
    parser.add_argument("output_file_path", help="output filename")
    args = parser.parse_args()
    in_name = args.input_file_path
    out_name = args.output_file_path
    main(in_name, out_name)
