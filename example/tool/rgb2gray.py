"""
[What] input a folder with RGB images, output a folder with 3-channel gray images
[Why] SBR dataset is gray-depth based, so that we would like to pretrain a model which is trained on 3-channel gray based images
"""
import os
import cv2
import numpy as np
import argparse
import glob


def gray_to_3_channel(img_gray):
    h, w = img_gray.shape
    img_gray_3_channel = np.zeros((h, w, 3))
    img_gray_3_channel[:,:,0] = img_gray[:,:]
    img_gray_3_channel[:,:,1] = img_gray[:,:]
    img_gray_3_channel[:,:,2] = img_gray[:,:]
    return img_gray_3_channel

def main(args):
    image_filenames = glob.glob(os.path.join(args.src_dir, "*.jpg"))
    for input_filename in image_filenames:
        color_image = cv2.imread(input_filename)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        output_image = gray_to_3_channel(gray_image)
        output_filename = os.path.join(args.dest_dir, os.path.basename(input_filename))
        cv2.imwrite(output_filename, output_image)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, type=str, default="/datasets/coco_2017/val2017")
    parser.add_argument('--dest_dir', required=True, type=str, default="/datasets/coco_2017/val2017_gray")
    args = parser.parse_args()
    main(args)
