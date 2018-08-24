import cv2
import numpy as np

from util.misc import bbox_area


def detect_objects(image_np, sess, detection_graph, min_score_thresh=0, max_num_detection=100):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # filter out too low score
    idxs = [k for k, v in enumerate(scores[0]) if v > min_score_thresh]

    # sort by self defined confidence(box_area*score) from large to small
    idxs = sorted(idxs, key=lambda i: -bbox_area(boxes[0][i])*scores[0][i])

    idxs = idxs[:max_num_detection]

    boxes = boxes[0][idxs]
    classes = classes[0][idxs]
    scores = scores[0][idxs]

    return boxes, classes, scores


