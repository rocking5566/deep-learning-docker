"""
[what] ignore all annotations in oid except the classes designated
[why] we would like to train on the designated classes dataset
"""
import os
import json
import argparse
import csv

def main(args):
    # 1. find out the labelname we desire
    label_map = {}
    with open(args.class_description_path) as f:
        reader = csv.reader(f)
        for row in reader:
            label_map[row[1]] = row[0]

    designated_labelnames = []
    for name in args.category_names:
        designated_labelnames.append(label_map[name])

    # 2. prepare rows to write (filter out bboxes we do not desire)
    ready_to_write_rows = []
    with open(args.src_annotation_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["LabelName"] not in designated_labelnames:
                continue
            # filter out source from not drawn by human
            if row["Source"] == "activemil":
                continue
            # filter out bboxes are group of
            if row["IsGroupOf"] == "1":
                continue
            # filter out bboxes are depiction
            if row["IsDepiction"] == "1":
                continue
            ready_to_write_rows.append(row)

    with open(args.dest_annotation_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=ready_to_write_rows[0].keys())
        writer.writeheader()
        for row in ready_to_write_rows:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_names", nargs='+', type=str, default=['Human hand'])
    parser.add_argument("--class_description_path", type=str, required=True)
    parser.add_argument("--src_annotation_path", type=str, required=True)
    parser.add_argument("--dest_annotation_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
