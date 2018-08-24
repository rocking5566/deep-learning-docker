"""
[what] ignore all annotations in coco except pedestrian and vehicle annotations
[why] we would like to train a pedestrian and vehicle(total 6 categories) only model
"""
import os
import json
import argparse

from pycocotools.coco import COCO

def reorder_category_id(cats, anns):
    # build mapping
    mapping = {}
    for c_idx, cat in enumerate(cats):
        mapping[cat['id']] = c_idx+1

    for cat in cats:
        cat['id'] = mapping[cat['id']]
    for ann in anns:
        ann['category_id'] = mapping[ann['category_id']]

    return cats, anns

def main(args):
    coco = COCO(args.src_annotation_path)

    # get categories whose name is one of assigned ones
    catIds = coco.getCatIds(catNms=args.category_names)
    cats = coco.loadCats(catIds)

    # get images whose annotation include one of catIds
    imgIds = []
    for catId in catIds:
        imgIds.extend(coco.getImgIds(catIds=[catId]))
    imgIds = set(imgIds)
    imgs = coco.loadImgs(imgIds)

    # get annotations whose category is one of catIds
    annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    if args.reorder_category_id:
        cats, anns = reorder_category_id(cats, anns)

    # output new annotation file
    output_json = {
        "images": imgs,
        "annotations": anns,
        "categories": cats,
    }

    os.makedirs(os.path.dirname(args.dest_annotation_path), exist_ok=True)
    with open(args.dest_annotation_path, "w") as f:
        json.dump(output_json, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reorder_category_id", action='store_true')
    parser.add_argument("--category_names", nargs='+', type=str, default=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])
    parser.add_argument("--src_annotation_path", type=str, required=True)
    parser.add_argument("--dest_annotation_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
