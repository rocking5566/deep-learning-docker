import numpy as np


# bbox format:
# np.array([ymin, xmin, ymax, xmax]) (0~1)
# scale: scalar or list[up_scale, down_scale, left_scale, right_scale]
def scale_bbox(bbox, scale=1.):
    ymin, xmin, ymax, xmax = bbox
    if isinstance(scale, list):
        up_s, down_s, left_s, right_s = scale
    elif isinstance(scale, float):
        up_s, down_s, left_s, right_s = scale, scale, scale, scale
    else:
        raise
    w = xmax - xmin
    h = ymax - ymin
    new_xmin = max(0, xmin - w*(left_s - 1.0)/2)
    new_ymin = max(0, ymin - h*(up_s - 1.0)/2)
    new_xmax = min(1, xmax + w*(right_s - 1.0)/2)
    new_ymax = min(1, ymax + h*(down_s - 1.0)/2)
    return np.array([new_ymin, new_xmin, new_ymax, new_xmax])

def scale_bboxes(bboxes, scale=1):
    new_bboxes = []
    for i, b in enumerate(bboxes):
        new_bboxes.append(scale_bbox(b, scale))
    return np.array(new_bboxes)

# bbox format:
# np.array([ymin, xmin, ymax, xmax])
def bbox_area(bbox):
    ymin, xmin, ymax, xmax = bbox
    w, h = abs(xmax - xmin), abs(ymax - ymin)
    return w*h

# bbox format:
# np.array([ymin, xmin, ymax, xmax])
def bbox_center(bbox):
    ymin, xmin, ymax, xmax = bbox
    y = (ymin+ymax)/2
    x = (xmin+xmax)/2
    return np.array([y, x])
