import csv
from collections import defaultdict
from bounding_box import GroundTruthBoundingBox, PredictBoundingBox


# TODO: abstract the parser, so that fit various data format

def ground_truth_bbox_parser(gt_bbox_path):
    """
        Args:
            gt_bbox_path: the path of ground truth bounding box file
        Returns:
            gt_bboxes_by_id[image_id, class_id]: array of ground truth bounding box classified by image ID and class ID
    """
    gt_bboxes_by_id = defaultdict(list)
    with open(gt_bbox_path) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            gt_bbox = GroundTruthBoundingBox(
                image_id=row[0],
                class_id=row[5],
                xmin=row[1],
                ymin=row[3],
                xmax=row[2],
                ymax=row[4]
            )
            gt_bboxes_by_id[gt_bbox.image_id, gt_bbox.class_id].append(gt_bbox)
    return gt_bboxes_by_id

def predict_bbox_parser(predict_bbox_file):
    """
        Args:
            predict_bbox_file: the path of predict bounding box file
        Returns:
            predict_bboxes_by_id[image_id, class_id]: array of predict bounding box classified by image ID and class ID
    """
    predict_bboxes_by_id = defaultdict(list)
    with open(predict_bbox_file) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            predict_bbox = PredictBoundingBox(
                image_id=row[0],
                class_id=row[1],
                conf=row[2],
                xmin=row[3],
                ymin=row[4],
                xmax=row[5],
                ymax=row[6]
            )
            predict_bboxes_by_id[predict_bbox.image_id, predict_bbox.class_id].append(predict_bbox)
    return predict_bboxes_by_id

