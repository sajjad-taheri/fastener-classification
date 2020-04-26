from math import atan2
from math import pi, degrees, cos, sin, radians
import cv2.cv2 as cv2
import numpy as np
from .orientation_v2 import get_orientation, draw_axis
import sys
sys.path.append("...")
from classification import boltclassifier
import config
from datetime import datetime


def get_cropped_pic(img_path, min_area, offsetWidth, offsetHeight, show_binary_image, show_original_image, model):
    """
    @Params
    min_area: if the detected contour is smaller than this, it shouldn't be counted.
    offsetWidth and Height: how many pixes do you want to add to the bounding box.
    show xxx: show the image in opencv.
    @Returns
    1. results_array: all the cropped images
    2. image: original image.
    """
    results = []
    position = []
    origin_img = cv2.imread(img_path)
    img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape

    # use filter to make the image blur.
    gray = cv2.bilateralFilter(img, 9, 75, 75)
    # threshold image, so it's easier to be detect contour.
    _, threshold_img = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
    # show the image which is preprocessed
    if show_binary_image:
        cv2.imshow('Gray image', cv2.resize(threshold_img, (w//2,h//2)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # find contours and get the external one
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # counter of contours
    index = 0
    for c in contours:
        # get the min area rect
        rect = cv2.minAreaRect(c)

        # get the width and height of minRec
        width = int(rect[1][0])
        height = int(rect[1][1])
        # make sure the area of rectangle is big enough
        # test if the contour is a bolt:
        if width * height > min_area:
            _, (length, height3), angle3 = rect
            if height3 > length:
                length = height3
            length = length / 6.6

            rect_with_offset = make_rect_bigger(rect, offsetWidth, offsetHeight)
            box = cv2.boxPoints(rect_with_offset)
            # cv2.putText(origin_img, str(index), tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX,
            #             5, (0, 255, 0), 2, cv2.LINE_AA)

            # convert all coordinates floating point values to int
            box = np.int0(box)
            src_pts = box.astype("float32")

            # get the new width and height
            width = int(rect_with_offset[1][0])
            height = int(rect_with_offset[1][1])
            # corrdinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[0, height],
                                [0, 0],
                                [width, 0],
                                [width, height]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(170, 170, 170))

            temp_name = config.TEMP_PATH + str(datetime.now().strftime('%Y%m%d-%H%M%S')) + '.jpg'
            cv2.imwrite(temp_name, warped)

            if model_predict(temp_name, model) == 0:
                bolt_name = config.BOLT_PATH + str(datetime.now().strftime('%Y%m%d-%H%M%S')) + '.jpg'
                cv2.imwrite(bolt_name, warped)
                # get the box points from original rectangle
                box = cv2.boxPoints(rect)
                # convert all coordinates floating point values to int
                box = np.int0(box)
                src_pts = box.astype("float32")
                # get the new width and height
                width = int(rect[1][0])
                height = int(rect[1][1])
                # coordinate of the points in box points after the rectangle has been straightened
                dst_pts = np.array([[0, height],
                                    [0, 0],
                                    [width, 0],
                                    [width, height]], dtype='float32')
                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                # directly warp the rotated rectangle to get the straightened rectangle
                warped = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(170, 170, 170))
                # get the center and angle.
                angle, center, evec1, evec2, eval = get_orientation(c, origin_img)
                # use angle but not radian.
                angle1 = angle * 180.0 / pi
                grasp_point = []
                # from the rect center to the centroid.
                angle2 = atan2(center[1] - int(rect[0][1]), center[0] - int(rect[0][0])) * 180.0 / pi
                direction_point = (center[0] + 0.02 * evec1 * eval,
                                   center[1] + 0.02 * evec2 * eval)
                real_angle = angle1
                # the angle between two vector should smaller than 90, otherwise we had a reversed direction.
                if min((360.0 - abs(angle1 - angle2)), abs(angle1 - angle2)) > 90.0:
                    real_angle = angle1 + 180.0
                    if real_angle > 180.0:
                        real_angle = real_angle - 360.0
                    direction_point = (center[0] - 0.02 * evec1 * eval,
                                       center[1] - 0.02 * evec2 * eval)
                draw_axis(origin_img, center, direction_point, (0, 255, 0), 1)
                # grasp_point.append((center[0] - 58) / 5.76)
                # grasp_point.append((center[1] - 111) / 5.63)
                # cv2.arrowedLine(origin_img, (int(rect[0][0]), int(rect[0][1])), center, (0, 0, 255),
                #                 thickness=1, line_type=8, shift=0, tipLength=5)

                results.append(bolt_name)
                # cv2.imshow("output", cv2.resize(output, (800,800)))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # draw a green 'nghien' rectangle
                # cv2.drawContours(origin_img, [box], 0, (0, 255, 0))
                index += 1

                # calculate head point
                length = max(height, width)
                angle_radians = radians(real_angle)
                xLen = cos(angle_radians) * length / 2 * 0.7
                yLen = sin(angle_radians) * length / 2 * 0.7
                head = (int(center[0] + xLen), int(center[1] + yLen))
                cv2.circle(origin_img, head, 2, (0, 255, 0), thickness=5)

                real_angle = degrees(real_angle) % 360

                position.append([real_angle, center, length, head])

            else:
                cv2.imwrite(config.NOT_BOLT_PATH + str(datetime.now().strftime('%Y%m%d-%H%M%S')) + '.jpg', warped)

    # draw contours of the image
    cv2.drawContours(origin_img, contours, -1, (0, 0, 255), 1)
    # check the image
    if show_original_image:
        cv2.imshow("contours", cv2.resize(origin_img, (w//2, h//2)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return position, results


def make_rect_bigger(rect, w, h):
    return (rect[0][0], rect[0][1]), (rect[1][0] + w, rect[1][1] + h), rect[2]


def model_predict(contour, model):
    return boltclassifier.classify(contour, model)


# if __name__ =='__main__':
#     bolt_path = "C:/SPARCS/code/app/output_images/conveyor/20191128-102407.jpg"
#     _y = get_cropped_pic(bolt_path, 1000, 20, 20, False, True, None)
# index = 0
# for imagefile in os.listdir(bolt_path):
#     index += 1
#     image_path = os.path.join(bolt_path, imagefile)
#     _y = get_cropped_pic(image_path, 1000, 20, 20, False, True, None)


# focal_length = 3.67
# f_x = 1515
# f_y = 1506
# m_x = 412.81  # pixel in mm
# m_y = 410.35
