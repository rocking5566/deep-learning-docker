import os
import sys
import numpy as np
from collections import defaultdict
from parser import ground_truth_bbox_parser, predict_bbox_parser

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

this_file_dirname = os.path.dirname(sys.argv[0])
GROUND_TRUTH_FILE = "/datasets/udacity_driving_datasets/labels_val.csv"
PREDICT_RESULT_FILE = os.path.join(
    this_file_dirname, "../results/ssd_udacity_driving_val.csv")


# step 1. read GT files and convert to our data structure
gt_bboxes_by_id = ground_truth_bbox_parser(GROUND_TRUTH_FILE)

# step 2. read result files and convert to our data structure
predict_bboxes_by_id = predict_bbox_parser(PREDICT_RESULT_FILE)

# step 3. run evaluation algorithm
def BBox_IoU(gt_bboxes, predict_bboxes):
    """
        Args:
            gt_bboxes: array of ground truth bounding boxes of single image, single class.
            predict_bboxes: array of predict bounding boxes of single image, single class.
        Returns:
            A two-dimension array (shape: len(predict_bboxes) by len(gt_bboxes), value: ratio of area of overlapped(0 ~ 1))
    """
    outputs = []

    # considering the case that no predict bbox
    if len(predict_bboxes) == 0:
        n = len(gt_bboxes)
        return np.array(outputs).reshape(0, n)

    for p_idx, predict_bbox in enumerate(predict_bboxes):
        output = []
        for gt_idx, gt_bbox in enumerate(gt_bboxes):
            xmin = max(predict_bbox.xmin, gt_bbox.xmin)
            ymin = max(predict_bbox.ymin, gt_bbox.ymin)
            xmax = min(predict_bbox.xmax, gt_bbox.xmax)
            ymax = min(predict_bbox.ymax, gt_bbox.ymax)
            w = (xmax-xmin)
            if w<0.0:
                w = 0
            h = (ymax-ymin)
            if h<0.0:
                h = 0
            area_inter = w*h
            area_outer = predict_bbox.area + gt_bbox.area - area_inter
            iou = area_inter/area_outer
            assert(iou>=0 and iou<=1), "IoU {} must equal or be greater than 0 and equal or be less than 1, area_inter: {}, area_outer: {}".format(iou, area_inter, area_outer)
            output.append(iou)
        outputs.append(output)
    return np.array(outputs)

def eval_IoU(IoUs, IoU_thres=0.5):
    """
        Args:
            IoUs: A two-dimension array (shape: len(predict_bboxes) by len(gt_bboxes), value: ratio of area of overlapped(0 ~ 1)).
            IoU_thres: a float value (0 ~ 1) of IoU threshold, closer to 0 means looser, and closer to 1 means more strict.
        Returns:
            A two-dimension array (shape: len(predict_bboxes) by len(gt_bboxes), value: match or not match(either 1 or 0))
    """
    outputs = []

    # considering the case that no predict bbox
    if len(IoUs) == 0:
        return np.array(outputs).reshape(IoUs.shape)

    gt_available_idx = [1]*len(IoUs[0])
    for p_idx, gt_ious in enumerate(IoUs):
        output = [0]*len(gt_ious)
        # considering the case that no ground truth bbox
        if len(output) == 0:
            return np.array(outputs).reshape(IoUs.shape)
        # find max iou
        gt_idx = np.argmax(gt_ious*gt_available_idx)
        # considering the case that gt_ious*gt_available_idx are all 0, must
        # have to double check the available slot
        if gt_available_idx[gt_idx]:
            if gt_ious[gt_idx] >= IoU_thres:
                output[gt_idx] = 1
                gt_available_idx[gt_idx] = 0
        outputs.append(output)
    return np.array(outputs)

match_matrices_by_id = defaultdict(list)
for (image_id, class_id) in predict_bboxes_by_id:
    # sort by conf(higher conf first serve)
    idxes = np.argsort([-bbox.conf for bbox in predict_bboxes_by_id[image_id, class_id] if bbox.conf >= CONFIDENCE_THRESHOLD])
    predict_bboxes = [predict_bboxes_by_id[image_id, class_id][i] for i in idxes]
    gt_bboxes = gt_bboxes_by_id[image_id, class_id]
    IoUs = BBox_IoU(gt_bboxes, predict_bboxes)
    # eval IoU by threshold
    match_matrices_by_id[image_id, class_id] = eval_IoU(IoUs, IoU_thres=IOU_THRESHOLD)

# step 4. output benchmark results
def summary_result(match_matrices_by_id):
    """
        Args:
            match_matrices_by_id: A two-dimension array (shape: len(predict_bboxes) by len(gt_bboxes), value: match or not match(1 or 0))
        Returns:
            results: A dict including precision, recall results
    """
    results = {}
    # accumulate all results
    num_predict_bbox = 0 # selected elements
    num_gt_bbox = 0 # relevant elements
    num_matched_bbox = 0 # True positive
    for (image_id, class_id) in match_matrices_by_id:
        match_matrix = match_matrices_by_id[image_id, class_id]
        num_p, num_gt = match_matrix.shape
        num_predict_bbox += num_p
        num_gt_bbox += num_gt
        num_matched_bbox += np.sum(match_matrix)
    results["precision"] = num_matched_bbox / num_predict_bbox
    results["recall"] = num_matched_bbox / num_gt_bbox
    return results
print(summary_result(match_matrices_by_id))
