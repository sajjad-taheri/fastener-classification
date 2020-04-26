import socket
from camera.camera_capture import CameraCapture
from processing.direction.crop_image_orientation import get_cropped_pic
import config
import tensorflow as tf
from classification.finefrainedclassifer import classify
from processing.projection import get_perspective_transformation, transform_point

print('[INFO] loading models...')
binary_model = tf.keras.models.load_model(config.BINARY_MODEL)
finegrained_model = tf.keras.models.load_model(config.FINEGRAINED_MODEL)

M = get_perspective_transformation()


# conveyor_image = CameraCapture.camera1_frame_capture(config.PATH)
conveyor_image = '/Users/Sajjad/Downloads/conveyor/20191209-144923.jpg'

# Positions are like [real_angle, center, length, head] with center[0] and center[1]
# results are the cropped images
positions, results = get_cropped_pic(conveyor_image, 900, 80, 80, False, False, binary_model)

if results:
    dst = transform_point(positions[0][1], M)
    xCoord = dst[0]
    yCoord = dst[1]
    rotation = positions[0][0]
    length = positions[0][2]
    pickPointHead = [positions[0][3][0], positions[0][3][1]]
else:
    xCoord = -1
    yCoord = -1
    rotation = -1
    length = -1
    pickPointHead = -1
print('[INFO] xCoord: {}, yCoord: {}, rotation: {}, length: {}, head: {}'.format(xCoord, yCoord, rotation,
                                                                                 length, pickPointHead))

bolt_number = classify([results[0]], finegrained_model)

