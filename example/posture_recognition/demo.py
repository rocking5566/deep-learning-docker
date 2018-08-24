import cv2
import argparse
import numpy as np
import time
import tensorflow as tf

import sys
sys.path.append("../../..")
from util.detection import detect_objects
from util.misc import scale_bboxes

LABELS = ['others', 'lie', 'sit', 'stand']


def preprocessing(frame_rgb):
    frame_rgb = cv2.resize(frame_rgb, (224, 224))
    frame_rgb = (frame_rgb / 127.5) - 1
    frame_rgb = frame_rgb.reshape(1, 224, 224, 3)
    return frame_rgb


def predict_posture(sess, tf_graph, img):
    img = preprocessing(img)
    img_tensor = tf_graph.get_tensor_by_name('input_1:0')
    classes = tf_graph.get_tensor_by_name('fc_predict/Softmax:0')
    pred = sess.run(classes, feed_dict={img_tensor: img})[0]
    cls_idx = pred.argmax(axis=0)
    confidence = pred[cls_idx]
    label = LABELS[cls_idx]
    return label, confidence


def load_tf_graph(model_name):
    with tf.gfile.GFile(model_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph


def main(args):
    # init detection graph
    detection_tf_graph = load_tf_graph(args.detection_pb)
    # init classification graph
    classification_tf_graph = load_tf_graph(args.classification_pb)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print('No video camera found')
        exit()

    print("OpenCV version :  {0}".format(cv2.__version__))

    with tf.Session(graph=classification_tf_graph) as classification_sess:
        with tf.Session(graph=detection_tf_graph) as detection_sess:
            while True:
                ret, frame_bgr = video_capture.read()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_h, img_w, _ = frame_rgb.shape

                # do detection
                t1 = cv2.getTickCount()
                boxes, classes, scores = detect_objects(
                    frame_rgb,
                    detection_sess,
                    detection_tf_graph,
                    args.detection_min_score_thresh,
                    args.detection_max_num_bbox)
                t2 = cv2.getTickCount()
                print("detection time:", (t2 - t1) / cv2.getTickFrequency())

                # enlarge bboxes a little
                boxes = scale_bboxes(boxes, scale=1.2)

                # do classification
                for box, cls, score in zip(boxes, classes, scores):
                    # crop
                    ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
                    ymin = int(ymin * img_h)
                    xmin = int(xmin * img_w)
                    ymax = int(ymax * img_h)
                    xmax = int(xmax * img_w)
                    cropped_img = frame_rgb[ymin:ymax, xmin:xmax]
                    # cv2.imshow('Cropped', cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                    t1 = cv2.getTickCount()
                    label, conf = predict_posture(classification_sess, classification_tf_graph, cropped_img)
                    t2 = cv2.getTickCount()
                    print("classification time:", (t2 - t1) / cv2.getTickFrequency())

                    # visualize
                    cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (255, 0, 255))
                    cv2.putText(frame_bgr, label+":"+str(conf), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

                cv2.imshow('Video', frame_bgr)
                # Press q to exit the program
                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord('q'):
                    break
            video_capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_pb', type=str, required=True)

    parser.add_argument('--detection_pbtxt', type=str, required=True)
    parser.add_argument('--detection_pb', type=str, required=True)
    parser.add_argument('--detection_num_classes', type=int, required=True)
    parser.add_argument('--detection_min_score_thresh', type=float, default=0.6)
    parser.add_argument('--detection_max_num_bbox', type=int, default=1)
    args = parser.parse_args()
    main(args)
