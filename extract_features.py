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
from skimage.segmentation import flood, flood_fill
from skimage.color import rgb2gray, gray2rgb
from skimage import measure
from skimage.feature import match_template
import math
import os
import helpers
import preprocessing
from scipy.signal import convolve2d
from scipy import ndimage


def getGradients(_img_bin, no_of_features=5):
    # print(_img.shape)
    try:
        img = np.copy(_img_bin)
        sy = filters.sobel_h(img)+0.0000000000000000000000000000000000001
        sx = filters.sobel_v(img)+0.0000000000000000000000000000000000001
        z = np.rad2deg(np.arctan(sy/sx))
        z = z.astype(int)
        # -180 -135 -90 -45 0 45 90 135 180
        n_bins = 9
        #delta_angle = 180//n_bins
        hist = np.zeros(n_bins)
        for k in range(n_bins):
            upper_bound = (k+1)*45 - 180
            lower_bound = k*45 - 180
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if z[i][j] < upper_bound and z[i][j] >= lower_bound:
                        hist[k] += 1

        hist /= (img.shape[0]*img.shape[1])
        hist *= 100
        return hist
    except Exception as ex:
        print("error image")
        print(img.shape)
        print(ex)
        return np.zeros(9)


def getHPP(_img_gray):
    img = np.copy(_img_gray)
    n_bins = 10
    height = 50
    img = resize(img, (height, height),
                 anti_aliasing=True)
    img = preprocessing.binarization(img)
    img = img == False
    horz_sum = np.sum(img, axis=1)
    f = np.zeros(n_bins)
    for i in range(n_bins):
        for j in range(int(height//n_bins)):
            f[i] += horz_sum[i*int(height//n_bins)+j]
    f /= (height*height)
    f *= 100
    return f


def getLPQ(_img_gray):
    winSize = 3
    img = np.copy(_img_gray)
    STFTalpha = 1 / winSize
    convmode = 'valid'
    img = np.float64(img)
    r = (winSize - 1) / 2
    x = np.arange(-r, r + 1)[np.newaxis]

    w0 = np.ones_like(x)
    w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
    w2 = np.conj(w1)

    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])

    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]
    LPQdesc = LPQdesc / LPQdesc.sum()
    # LPQdesc *= 100
    return LPQdesc


def getSDs(_img_gray):
    img = _img_gray.copy()
    binary = preprocessing.binarization(img)
    horizontal_hist = np.sum(binary, axis=1)
    img_with_Diacritics = np.copy(binary)
    img_with_Diacritics = img_with_Diacritics.astype(np.uint8)
    basline = np.argmin(horizontal_hist)
    seed = []
    temp = img_with_Diacritics[basline, 0]
    cp = gray2rgb(img_with_Diacritics)
    for i in range(1, len(img_with_Diacritics[basline])):
        if temp == 1 and img_with_Diacritics[basline, i] == 0:
            seed.append((i, basline))
        temp = img_with_Diacritics[basline, i]
    for i in seed:
        cv2.floodFill(img_with_Diacritics, None, i, 255)
    image_without_diacritic = (binary-img_with_Diacritics)*-1
    f_LATER_TO_DO = []
    return image_without_diacritic, f_LATER_TO_DO


def getWOr(_img_gray):
    image_without_diacritic, _ = getSDs(_img_gray)
    image_without_diacritic = image_without_diacritic.astype(np.uint8)
    contour = measure.find_contours(image_without_diacritic)
    s = 0
    for c in contour:
        rows, cols = c.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        x_axis = np.array([1, 0])
        line = np.array([vx, vy])
        dot_product = np.dot(x_axis, line)
        angle_2_x = np.arccos(dot_product)
        angle = math.degrees(angle_2_x)
        s += angle
    if (len(contour) != 0):
        s = s/len(contour)
    else:
        s = 0.0
    return [s]


def getLVL(_skeleton_img, lineThresholdFraction=0.1):
    skeleton_img = np.copy(_skeleton_img)
    skeleton_img = skeleton_img == False
    rows, cols = skeleton_img.shape
    featureVector = []
    verticalLines = []

    for col in range(cols):
        cnt = 0
        tempCnt = 0
        minl, maxr = 1e9, 0

        for row in range(rows):
            if skeleton_img[row, col] == 1:
                tempCnt += 1
                minl = min(minl, row)
                maxr = max(maxr, row)
            else:
                if tempCnt > cnt:
                    cnt = tempCnt
                tempCnt = 0
        if cnt > rows*lineThresholdFraction:
            verticalLines.append(cnt)

    if len(verticalLines) == 0:
        verticalLines.append(0)
    verticalLines = np.array(verticalLines)

    featureVector.append(np.max(verticalLines)/rows)  # longest line normalized
    # ratio between longest line and text height
    featureVector.append(np.max(verticalLines)/(maxr-minl+1))
    featureVector.append(np.var(verticalLines))  # variance among lines
    return featureVector


def getTTH(_skeleton_img, minThicknessThreshold=5, maxThicknessThreshold=100):
    skeleton_img = np.copy(_skeleton_img)
    skeleton_img = skeleton_img == False
    rows, cols = skeleton_img.shape
    allThickness = []
    for row in range(rows):
        tl, tr = -1, -1
        for col in range(cols):
            if skeleton_img[row, col] == 1:
                if tl == -1:
                    tl = col
                else:
                    tr = col
            else:
                if tr-tl+1 > minThicknessThreshold and tr-tl+1 < maxThicknessThreshold:
                    allThickness.append(tr-tl+1)
                tl, tr = -1, -1
    if len(allThickness) == 0:
        return [0]
    return [np.mean(allThickness)]


def getHVSL(_gray_img, isTextBlack):
    # _, bw = cv2.threshold(_gray_img, 150, 255, cv2.THRESH_BINARY)
    bw = preprocessing.binarization(_gray_img, isTextBlack)
    bw = np.uint8(bw)*255  # 0 ,1
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    img_sk = preprocessing.skeletonization(bw)
    img_sk = img_sk == False
    # helpers.show_images([img_sk])
    img_sk = np.uint8(img_sk)*255
    horizontal = np.copy(edges)
    vertical = np.copy(edges)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = max(cols // 30, 2)
    # print(horizontal_size)
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = max(rows // 30, 2)
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    num_labelsV, _ = cv2.connectedComponents(vertical)
    num_labelsH, _ = cv2.connectedComponents(horizontal)
    feature = num_labelsV/num_labelsH
    return [feature]
