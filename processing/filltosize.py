import cv2


class FillToSizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (h, w) = image.shape[:2]

        half_width = int((self.width-w)/2)
        half_height = int((self.height-h)/2)
        if (half_width * 2) + w == self.width:
            if (half_height * 2) + h == self.height:
                image = cv2.copyMakeBorder(image, top=int((self.height-h)/2), bottom=int((self.height-h)/2),
                                           left=int((self.width-w)/2), right=int((self.width-w)/2),
                                           borderType=cv2.BORDER_CONSTANT, value=(150, 150, 150))
            else:
                image = cv2.copyMakeBorder(image, top=int((self.height - h) / 2)+1, bottom=int((self.height - h) / 2),
                                           left=int((self.width - w) / 2), right=int((self.width - w) / 2),
                                           borderType=cv2.BORDER_CONSTANT, value=(150, 150, 150))
        else:
            if (half_height * 2) + h == self.height:
                image = cv2.copyMakeBorder(image, top=int((self.height-h)/2), bottom=int((self.height-h)/2),
                                           left=int((self.width-w)/2)+1, right=int((self.width-w)/2),
                                           borderType=cv2.BORDER_CONSTANT, value=(150, 150, 150))
            else:
                image = cv2.copyMakeBorder(image, top=int((self.height - h) / 2)+1, bottom=int((self.height - h) / 2),
                                           left=int((self.width - w) / 2)+1, right=int((self.width - w) / 2),
                                           borderType=cv2.BORDER_CONSTANT, value=(150, 150, 150))

        return image
