import cv2
import numpy as np
import time
from keras.models import load_model
import tensorflow as tf

skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
labels = ['Five_Finger', 'Four_Finger', 'Good_Sign', 'One_Finger', 'Three_Finger', 'Two_Finger',  'other']


def skin_mask(frame, x0, y0, width, height):
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    # bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res


def predict_gesture(sess, tf_graph, img):
    img = img.reshape(1, 200, 200, 1)
    # TODO - refine layer name
    img_tensor = tf_graph.get_tensor_by_name('input_1_1:0')
    classes = tf_graph.get_tensor_by_name('activation_4_1/Softmax:0')
    pred = sess.run(classes, feed_dict={img_tensor: img})[0]
    return labels[pred.argmax(axis=0)]


def load_tf_graph(model_name):
    with tf.gfile.GFile('gesture_model.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    for op in graph.get_operations():
        print(op.name)

    return graph


if __name__ == '__main__':
    height = 200
    width = 200
    x0 = 350
    y0 = 230
    tf_graph = load_tf_graph('gesture_model.hdf5')

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print('No video camera found')
        exit()

    print("OpenCV version :  {0}".format(cv2.__version__))

    with tf.Session(graph=tf_graph) as sess:
        while True:
            ret, frame = video_capture.read()
            roi = skin_mask(frame, x0, y0, width, height)

            t1 = cv2.getTickCount()
            pred = predict_gesture(sess, tf_graph, roi)
            t2 = cv2.getTickCount()
            print((t2 - t1) / cv2.getTickFrequency())
            cv2.putText(frame, pred, (x0 - 100, y0 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            cv2.imshow('Video', frame)
            cv2.imshow('roi', roi)
            # Press q to exit the program
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('i'):
                y0 = y0 - 5
            elif key == ord('k'):
                y0 = y0 + 5
            elif key == ord('j'):
                x0 = x0 - 5
            elif key == ord('l'):
                x0 = x0 + 5
            elif key == ord('r'):
                cv2.imwrite('5_' + str(time.time()) + '.jpg', roi)
            elif key == ord('t'):
                cv2.imwrite('test.jpg', roi)

    video_capture.release()
