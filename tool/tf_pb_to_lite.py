import os
import argparse

import tensorflow as tf


def main(args):
    graph_def_file = args.in_pb
    input_arrays = args.input_layers.split(",")
    output_arrays = args.output_layers.split(",")

    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
      graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    with open(args.out_lite, "wb") as f:
        f.write(tflite_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_pb', type=str, required=True)
    parser.add_argument('--out_lite', type=str, required=True)
    parser.add_argument('--input_layers', type=str, required=True)
    parser.add_argument('--output_layers', type=str, required=True)
    args = parser.parse_args()
    main(args)
