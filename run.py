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

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('localhost', 9001)
print('[INFO] Starting up on port {}'.format(server_address))
sock.bind(server_address)

sock.listen(1)

M = get_perspective_transformation()

while True:
    print('[INFO] Waiting for a connection ...')
    connection, client_address = sock.accept()

    try:
        print('[INFO] Connection from {}'.format(client_address))
        results = None

        while True:
            data = connection.recv(16)
            print('[INFO] Received {}'.format(data))
            if data:
                strData = data.decode('ASCII')
                if 'StartBand' in strData:
                    conveyor_image = CameraCapture.camera1_frame_capture(config.PATH)

                    # Positions are like [real_angle, center, length, head] with center[0] and center[1]
                    # results are the cropped images
                    positions, results = get_cropped_pic(conveyor_image, 900, 80, 80, False, False, binary_model)

                    if results:
                        dst = transform_point(positions[0][1], M)
                        xCoord = dst[0]
                        yCoord = dst[1]
                        rotation = positions[0][0]
                        length = positions[0][2]
                        pickPointHead = positions[0][3]
                    else:
                        xCoord = -1
                        yCoord = -1
                        rotation = -1
                        length = -1
                        pickPointHead = -1
                    print('[INFO] xCoord: {}, yCoord: {}, rotation: {}'.format(xCoord, yCoord, rotation))
                    connection.sendall(str.encode('' + str(xCoord) + ';' + str(yCoord) + ';' + str(rotation) + ';' +
                                                  str(length) + ';0;' + str(pickPointHead)))
                if 'StartFutter' in strData:
                    chuck_image = CameraCapture.camera2_frame_capture(config.SECOND_PATH)
                    bolt_number = classify([chuck_image, results[0]], finegrained_model)

                    connection.sendall(str.encode('S'+str(bolt_number)))
            else:
                print('[INFO] No more data from {}'.format(client_address))
                break

    finally:
        connection.close()
