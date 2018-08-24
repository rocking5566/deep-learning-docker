"""
[what] Annotate people image as coco format
[why] Prepare to classify posture
"""
import sys

sys.path.append("/opt/tf_model/research")

import os
import cv2
import argparse
import json
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append("../../../")
from util.detection import detect_objects
from util.misc import scale_bboxes
from util.misc import scale_bbox

CATEGORIES = [
    {
      "name": "others",
      "id": 0,
      "supercategory": "people"
    },
    {
      "name": "lie",
      "id": 1,
      "supercategory": "people"
    },
    {
      "name": "sit",
      "id": 2,
      "supercategory": "people"
    },
    {
      "name": "stand",
      "id": 3,
      "supercategory": "people"
    }
]
PERSON_CLASS_ID = 1

def _get_category_id_by_name(category_name):
    for c in CATEGORIES:
        if c["name"] == category_name:
            return c["id"]
    raise
def _get_category_name_by_id(category_id):
    for c in CATEGORIES:
        if c["id"] == category_id:
            return c["name"]
    raise


IMAGE_ID = 0
def _gen_image_id():
    global IMAGE_ID
    IMAGE_ID = IMAGE_ID + 1
    return IMAGE_ID

def _gen_image_json(img, filename):
    h, w, c = img.shape
    return {
        "width": w,
        "height": h,
        "file_name": filename,
        "data_captured": "2013-11-14 17:02:52",
        "id": _gen_image_id()
    }

BBOX_ID = 0
def _gen_bbox_id():
    global BBOX_ID
    BBOX_ID = BBOX_ID + 1
    return BBOX_ID

def _gen_bbox_json(box, img_w, img_h, category_id, image_id):
    # source box is [ymin, xmin, ymax, xmax] (0~1)
    return {
        "id": _gen_bbox_id(),
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [
            int(box[1] * img_w),
            int(box[0] * img_h),
            int((box[3] - box[1]) * img_w),
            int((box[2] - box[0]) * img_h)
        ] # convert to [x, y, w, h] (abs pixel)
    }

def init_annotation_file(annotation_filepath):
    if not os.path.exists(annotation_filepath):
        with open(annotation_filepath, "w") as f:
            output = {"categories": CATEGORIES, "images": [], "annotations": []}
            json.dump(output, f)

def image_exist(img_rel_path, annotation_filepath):
    with open(annotation_filepath, "r") as f:
        output = json.load(f)

    for img in output["images"]:
        if img["file_name"] == img_rel_path:
            return True
    return False

#Assume all people belong to same category in one image.
def annotate_image(img_name, img_bgr, category_id, annotation_filepath):
    with open(annotation_filepath, "r") as f:
        output = json.load(f)

    category_name = _get_category_name_by_id(category_id)

    # NOTE: this is not a good practice at all, but it is enough for our case
    # update IDs for _gen_image_json()
    global IMAGE_ID
    IMAGE_ID = max([img["id"] for img in output["images"]], default=0)

    # save image annotation to json file
    rel_image_filename = os.path.join(category_name, img_name)
    image_json = _gen_image_json(img_bgr, rel_image_filename)
    output["images"].append(image_json)
    with open(annotation_filepath, "w") as f:
        json.dump(output, f)

def annotate_bbox(img_bgr, bbox, category_id, annotation_filepath):
    with open(annotation_filepath, "r") as f:
        output = json.load(f)

    # NOTE: this is not a good practice at all, but again it is enough for our case
    # update IDs for _gen_bbox_json()
    global IMAGE_ID, BBOX_ID
    IMAGE_ID = max([img["id"] for img in output["images"]], default=0)
    BBOX_ID = max([ann["id"] for ann in output["annotations"]], default=0)

    # save bbox to json file
    height, width, channel = img_bgr.shape
    annotation_json = _gen_bbox_json(bbox, width, height, category_id, IMAGE_ID)
    output["annotations"].append(annotation_json)
    print(annotation_json)
    with open(annotation_filepath, "w") as f:
        json.dump(output, f)

def render_bbox_result(frame_rgb, boxes, classes, scores, category_index):
    show_rgb = frame_rgb.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
        show_rgb,
        boxes,
        classes.astype(np.int32),
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=0)
    show_bgr = cv2.cvtColor(show_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', show_bgr)

def render_roi(frame_rgb, bbox):
    ymin, xmin, ymax, xmax = bbox
    xmin = int(xmin * frame_rgb.shape[1])
    xmax = int(xmax * frame_rgb.shape[1])
    ymin = int(ymin * frame_rgb.shape[0])
    ymax = int(ymax * frame_rgb.shape[0])

    roi_rgb = frame_rgb[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('roi', roi_bgr)


def main(args):
    # init pbtxt
    label_map = label_map_util.load_labelmap(args.detection_pbtxt)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=args.detection_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # init misc
    init_annotation_file(args.annotation_filepath)
    category_id = _get_category_id_by_name(args.category_name)
    os.makedirs(os.path.dirname(args.annotation_filepath), exist_ok=True)
    category_name = _get_category_name_by_id(category_id)
    image_dir = os.path.join(args.image_dir, category_name)

    # load object detection model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.detection_pb, 'rb') as fid:
            serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        with tf.Session(graph=detection_graph) as sess:
            for img_name in os.listdir(image_dir):
                if image_exist(os.path.join(category_name, img_name), args.annotation_filepath):
                    continue
                print (img_name)
                img_full_name = os.path.join(image_dir, img_name)
                if os.path.isfile(img_full_name):
                    frame_bgr = cv2.imread(img_full_name)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # detect people
                    boxes, classes, scores = detect_objects(frame_rgb, sess, detection_graph,
                                                            args.detection_min_score_thresh)

                    # enlarge bbox a little to get more context
                    new_boxes = scale_bboxes(boxes, scale=1.2)

                    if new_boxes.size == 0:
                        continue

                    annotate_image(img_name, frame_bgr, category_id, args.annotation_filepath)

                    # visualize it
                    for i, bbox in enumerate(new_boxes):
                        if classes[i] != PERSON_CLASS_ID:
                            continue

                        render_bbox_result(frame_rgb, new_boxes, classes, scores, category_index)
                        render_roi(frame_rgb, bbox)

                        pressed_key = cv2.waitKey(0)
                        if pressed_key & 0xFF == ord('q'):
                            return
                        elif pressed_key & 0xFF == ord('n'):
                            annotate_bbox(frame_bgr, bbox, category_id, args.annotation_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_filepath", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--category_name", type=str, required=True)

    parser.add_argument('--detection_pbtxt', type=str, required=True)
    parser.add_argument('--detection_pb', type=str, required=True)
    parser.add_argument('--detection_num_classes', type=int, required=True)
    parser.add_argument('--detection_min_score_thresh', type=float, default=0.6)
    args = parser.parse_args()
    main(args)
