import cv2
from datetime import datetime
import numpy as np


class CameraCapture:
    @staticmethod
    def camera1_frame_capture(path):
        cap = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 8)

        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = image[55: 1080, 250: 1800]

        pts1 = np.float32(
            [[223, 55],
             [110, 865],
             [1349, 33],
             [1542, 828]]
        )
        pts2 = np.float32(
            [[223, 55],
             [223, 865],
             [1349, 55],
             [1349, 865]]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        crop = cv2.warpPerspective(crop, M, (1550, 1025), borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(170, 170, 170))

        image_name = path + str(datetime.now().strftime('%Y%m%d-%H%M%S')) + '.jpg'

        print('[INFO] writing the image {}'.format(image_name))
        cv2.imwrite(image_name, crop)

        cap.release()
        cv2.destroyAllWindows()

        return image_name

    @staticmethod
    def camera2_frame_capture(path):
        cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 20)

        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = image[475: 725, 785: 1035]
        image_name = str(datetime.now().strftime('%Y%m%d-%H%M%S')) + '.jpg'
        image_name = path + image_name

        print('[INFO] writing the image {}'.format(image_name))
        cv2.imwrite(image_name, crop)

        cap.release()
        cv2.destroyAllWindows()

        return image_name
