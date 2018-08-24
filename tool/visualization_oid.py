import os
import argparse
import json
import csv
from collections import defaultdict

import cv2


def main(args):
    rows = []
    with open(args.annotation_path) as f:
        for row in csv.DictReader(f, delimiter=','):
            rows.append(row)
            ##if label != "/m/0k65p": # human hand
            ##    continue

            ##if row["Source"] != "freeform" or row["Source"] != "xclick":
            ##    continue

            ##if row["IsOccluded"] != "1":
            ##    continue

            ##if row["IsTruncated"] != "1":
            ##    continue

            ##if row["IsGroupOf"] != "1":
            ##    continue

            ##if row["IsDepiction"] != "1":
            ##    continue

            ##if row["IsInside"] != "1":
            ##    continue

    image_IDs = sorted(set([row["ImageID"] for row in rows]))

    imgid_labels_map = defaultdict(list)
    for row in rows:
        imgid_labels_map[row["ImageID"]].append(row)

    for img_ID in image_IDs:
        image_file = os.path.join(args.image_dir, img_ID+".jpg")
        img = cv2.imread(image_file)
        img_h, img_w, _ = img.shape

        for row in imgid_labels_map[img_ID]:
            ymin, xmin, ymax, xmax = float(row["YMin"]), float(row["XMin"]), float(row["YMax"]), float(row["XMax"])
            # update coordinate
            ymin, xmin, ymax, xmax = int(ymin*img_h), int(xmin*img_w), int(ymax*img_h), int(xmax*img_w)

            # draw bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255))

        cv2.imshow("demo", img)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
