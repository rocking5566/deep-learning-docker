import sys

sys.path.append("/opt/tf_model/research")

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append("../")
from util.detection import detect_objects


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

print("OpenCV version :  {0}".format(cv2.__version__))

# Should run under docker container from tensorflow_object_detection
ROOT = '/opt/tf_model/research/object_detection/'

# Download pre-train SSD-MobileNet model from
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

MODEL_ROOT = '/datasets/models/tf/'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(MODEL_ROOT, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(ROOT, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbtxt', type=str, default=PATH_TO_LABELS)
    parser.add_argument('--pb', type=str, default=PATH_TO_CKPT)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--min_score_thresh', type=float, default=0.5)
    parser.add_argument('--gray', action='store_true')
    parser.add_argument('--camera_zoom', type=float, default=1)
    parser.add_argument('--camera_size', type=str, default="640 480") # w, h
    parser.add_argument('--video_src', type=str, default=None)
    args = parser.parse_args()

    camera_w, camera_h = map(int, args.camera_size.split())
    if args.video_src is None:
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)
        assert video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) == camera_w, "Real camera width is set %d" % video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        assert video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == camera_h, "Real camera height is set %d" % video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        video_capture = cv2.VideoCapture(args.video_src)
    if not video_capture.isOpened():
        print('No video camera found')
        exit()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(args.pbtxt)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=args.num_classes, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)

    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = video_capture.read()
            if args.gray:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # central crop
            img_h, img_w, _ = frame_rgb.shape
            margin_ratio = (args.camera_zoom-1)/args.camera_zoom/2
            margin_w, margin_h = int(img_w*margin_ratio), int(img_h*margin_ratio)
            frame_rgb = frame_rgb[margin_h:img_h-margin_h, margin_w:img_w-margin_w]

            boxes, classes, scores = detect_objects(frame_rgb, sess, detection_graph, args.min_score_thresh)

            # Visualization of the results of a detection.
            result_rgb = frame_rgb.copy()
            vis_util.visualize_boxes_and_labels_on_image_array(
                result_rgb,
                boxes,
                classes.astype(np.int32),
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0)

            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', result_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
