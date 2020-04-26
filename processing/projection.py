import cv2.cv2 as cv2
import numpy as np
import config


def get_perspective_transformation():
    pts1 = np.float32(
        [[115, 866],
         [1549, 858],
         [1368, 60],
         [245, 60]]
    )
    pts2 = np.float32(
        [[0, 1080],
         [1550, 1080],
         [1550, 0],
         [0, 0]]
    )

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return M


def transform_point(point, M):
    point = np.array([1224, 590])
    homogeneous_point = [point[0], point[1], 1]
    homogeneous_point = M.dot(homogeneous_point)
    homogeneous_point /= homogeneous_point[2]
    result = homogeneous_point[:2] / config.PIXEL_TO_MM_RATIO

    return result
