import glob

import cv2
import numpy as np
from matplotlib import image as mpimg


def calibration():
    images = glob.glob("camera_cal/calibration*.jpg")

    imgpoints = []
    objpoints = []

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x, y coordinates

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def generate_warp_config():
    # source for corner points: https://github.com/js1972
    corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
    new_top_left = np.array([corners[0, 0], 0])
    new_top_right = np.array([corners[3, 0], 0])
    offset = [50, 0]

    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

    warp_matrix = cv2.getPerspectiveTransform(src, dst)
    warp_matrix_inverse = cv2.getPerspectiveTransform(dst, src)
    return warp_matrix, warp_matrix_inverse


def warp_image(image, warp_matrix):
    img_size = (image.shape[1], image.shape[0])

    warped = cv2.warpPerspective(image, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    return warped