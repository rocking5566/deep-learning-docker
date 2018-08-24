#!/usr/bin/python
import os
import cv2
import numpy as np
import time
from keras import backend as K
from keras.models import load_model

from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from ssd_box_encode_decode_utils import decode_y2


INPUT_CAM_WIDTH=1280
INPUT_CAM_HEIGHT=720
MODEL_INPUT_WIDTH=480
MODEL_INPUT_HEIGHT=300

CLASSES = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
# TODO: use a more graceful method to color bbox
COLORS = [(0,0,0), (0,0,255), (0,255,0), (255,0,0), (0,0,0), (0,0,0)] # emphasize car as Red, truck as Green, pedestrian as Blue
def plot_bboxes(img, Y_pred):
    for y in Y_pred:
        confidence_score = y[1]
        label_string = "{}: {:.2f}".format(CLASSES[int(y[0])], confidence_score)

        pt1 = (int(y[2]), int(y[3]))
        pt2 = (int(y[4]), int(y[5]))
        color = COLORS[int(y[0])]

        cv2.rectangle(img, pt1, pt2, color)
        cv2.putText(img, label_string, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    return img

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_CAM_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_CAM_HEIGHT)

    if not video_capture.isOpened():
        raise('No video camera found')

    # load keras model
    MODEL_PATH = "model/ssd7_udacity_driving.h5"

    # We need to create an SSDLoss object in order to pass that to the model
    # loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    model = load_model(MODEL_PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'compute_loss': ssd_loss.compute_loss})

    while(True):
        ret, frame = video_capture.read()

        # TODO: double check model input channel format, BGR or RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_rgb_frame = cv2.resize(rgb_frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        X = np.array([resized_rgb_frame])
        start_time = time.time()
        y_pred = model.predict(X)
        interval = time.time() - start_time
        print("prediction time:", interval)

        # Decode the raw prediction `y_pred`
        y_pred_decoded = decode_y2(y_pred,
                                   confidence_thresh=0.7,
                                   iou_threshold=0.4,
                                   top_k='all',
                                   input_coords='centroids',
                                   normalize_coords=True,
                                   img_height=INPUT_CAM_HEIGHT,
                                   img_width=INPUT_CAM_WIDTH)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        #print("Decoded predictions (output format is [class_id, confidence, xmin, ymin, xmax, ymax]):\n")
        #print(y_pred_decoded)

        frame = plot_bboxes(frame, y_pred_decoded[0])

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
