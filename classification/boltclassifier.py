import numpy as np
import cv2.cv2 as cv2
from tensorflow.keras.preprocessing.image import img_to_array


def classify(image, model):
    target_size = 299

    image = cv2.imread(image, 1)
    (h, w) = image.shape[:2]

    if h > w:
        image = cv2.resize(image, (int(w * target_size / h), target_size))
        image = cv2.copyMakeBorder(image, 0, 0, target_size - int(w * target_size / h), 0,
                                   cv2.BORDER_CONSTANT, value=(170, 170, 170))
    else:
        image = cv2.resize(image, (target_size, int(h * target_size / w)))
        image = cv2.copyMakeBorder(image, 0, target_size - int(h * target_size / w), 0, 0,
                                   cv2.BORDER_CONSTANT, value=(170, 170, 170))

    image = image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image).argmax(axis=1)
    print('[INFO] predicted: {}'.format(prediction))

    return prediction
