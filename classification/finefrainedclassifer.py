import numpy as np
import cv2
from processing.filltosize import FillToSizePreprocessor
from tensorflow.keras.preprocessing.image import img_to_array


def classify(images, model):
    offset_size = 700
    target_size = 500
    image = cv2.imread(images[0], 1)
    (h, w) = image.shape[:2]

    if h > offset_size:
        image = image[(h - offset_size)/2: offset_size + (h - offset_size)/2, 0: w]
    if w > offset_size:
        image = image[0: h, (w - offset_size)/2: offset_size + (w - offset_size)/2]

    fts = FillToSizePreprocessor(offset_size, offset_size)

    image = fts.preprocess(image)

    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image).argmax(axis=1)

    print('[INFO] predicted: {}'.format(prediction))

    return prediction

