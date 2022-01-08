import argparse
import helpers
import preprocessing
import extract_features
import cv2


def main(in_name, out_name):
    imgsPaths = helpers.getImgsPaths(in_name)
    for imgPath in imgsPaths:
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

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", help="input filename")
    parser.add_argument("output_file_path", help="output filename")
    args = parser.parse_args()
    in_name = args.input_file_path
    out_name = args.output_file_path
    main(in_name, out_name)
