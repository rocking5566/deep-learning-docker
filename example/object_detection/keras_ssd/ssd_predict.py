import os
import sys
import cv2
import math
import numpy as np
import csv

import keras
from keras import backend as K
from keras.models import load_model
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from ssd_box_encode_decode_utils import decode_y2

from network.mobilenetv1 import relu6
from network.mobilenetv1 import DepthwiseConv2D


MODEL_PATH = "model/ssd300_mobilenetv1_udacity_driving.h5"
RESULT_FILE = "results/ssd_udacity_driving_val.csv"

# 1. read keras model
this_file_dirname = os.path.dirname(sys.argv[0])
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model = load_model(
            os.path.join(this_file_dirname, MODEL_PATH),
            custom_objects={
                'AnchorBoxes': AnchorBoxes,
                'compute_loss': ssd_loss.compute_loss,
                'relu6': relu6,
                'DepthwiseConv2D': DepthwiseConv2D,
            }
        )

# 2. read evaluation list
test_image_dir = "/datasets/udacity_driving_datasets/"
list_file = "/datasets/udacity_driving_datasets/labels_val.csv"
test_image_files = []
with open(list_file) as f:
    reader = csv.reader(f, delimiter=",")
    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        test_image_file = row[0]
        test_image_files.append(test_image_file)
test_image_files = set(test_image_files)
print("total num of test image file", len(test_image_files))

# 2.a prepare data generator
def data_generator(test_image_files, test_image_dir, batch_size=1):
    n = len(test_image_files)
    num_iter = int(math.ceil(n/batch_size))

    def g(eles):
        for ele in eles:
            yield ele
    test_image_file_generator = g(test_image_files)

    for idx in range(num_iter):
        print("running iter", idx)
        X = []
        filenames = []
        for jdx in range(batch_size):
            try:
                img_file_name = next(test_image_file_generator)
            except StopIteration:
                break
            img_file_path = os.path.join(test_image_dir, img_file_name)
            x = cv2.imread(img_file_path)
            X.append(x)
            filenames.append(img_file_name)
        X = np.array(X)
        yield X, filenames
dg = data_generator(test_image_files, test_image_dir, batch_size=32)

# 3. predict
def predict(model, dg):
    for X, filenames in dg:
        Y_pred = model.predict(X)
        img_height, img_width = X[0].shape[:-1]
        Y_pred_decoded = decode_y2(
            Y_pred,
            confidence_thresh=0.5,
            iou_threshold=0.4,
            top_k='all',
            input_coords='centroids',
            normalize_coords=True,
            img_height=img_height,
            img_width=img_width)
        for idx, y_pred_decoded in enumerate(Y_pred_decoded):
            for bbox in y_pred_decoded:
                yield [filenames[idx]]+bbox.tolist()
result_bboxes = predict(model, dg)

# 4. write result file
with open(RESULT_FILE, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['image id', 'class id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    for box in result_bboxes:
        writer.writerow(box)
