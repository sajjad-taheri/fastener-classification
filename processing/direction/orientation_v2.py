from __future__ import print_function
from __future__ import division
import cv2 as cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi


def draw_axis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def get_orientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty(0)
    # mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    covar, mean = cv2.calcCovarMatrix(data_pts, mean, cv2.COVAR_SCALE |
                                      cv2.COVAR_ROWS |
                                      cv2.COVAR_SCRAMBLED)

    eVal, eVec = cv2.eigen(covar)[1:]

    # Conversion + normalisation required due to 'scrambled' mode
    eVec = cv2.gemm(eVec, data_pts - mean, 1, None, 0)
    # apply_along_axis() slices 1D rows, but normalize() returns 4x1 vectors
    eVec = np.apply_along_axis(lambda n: cv2.normalize(n, n).flat, 1, eVec)

    # Store the center of the object
    # cntr2 = (int(mean[0, 0]), int(mean[0, 1]))
    M = cv2.moments(pts)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cntr = (cX, cY)

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    # p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
    #       cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    # p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
    #       cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    p1 = (cntr[0] + 0.02 * eVec[0, 0] * eVal[0, 0],
          cntr[1] + 0.02 * eVec[0, 1] * eVal[0, 0])
    p2 = (cntr[0] - 0.02 * eVec[1, 0] * eVal[1, 0],
          cntr[1] - 0.02 * eVec[1, 1] * eVal[1, 0])
    # draw_axis(img, cntr, p1, (0, 255, 0), 1)
    # draw_axis(img, cntr, p2, (255, 255, 0), 5)
    # angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    angle = atan2(eVec[0, 1], eVec[0, 0])
    return angle, cntr, eVec[0, 0], eVec[0, 1], eVal[0, 0]


def calculate_orientation(image):
    src = cv2.imread(image)
    # Check if image is loaded successfully
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    cv2.imshow('src', src)
    # Convert image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV )
    _, contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore cnts that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
        # Draw each contour only for visualisation purposes
        cv2.drawContours(src, contours, i, (0, 0, 255), 2)
        # Find the orientation of each shape
        get_orientation(c, src)

        # # compute the center of the contour
        # M = cv2.moments(c)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        #
        # # draw the contour and center of the shape on the image
        # cv2.circle(src, (cX, cY), 7, (100, 255, 255), -1)
        # cv2.putText(src, "GraspPoint", (cX - 20, cY - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2)
    cv2.imshow('output', src)
    cv2.imwrite('Orientation.png', src)
    cv2.waitKey()


if __name__ == '__main__':
    calculate_orientation('/Users/Sajjad/Developer/NeuPro/SPARCS/research_code/bolt-frame-2019-10-30-17-04-34-678121-5.jpg')

