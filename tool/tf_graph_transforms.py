import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


def main(args):
    input_pb = args.in_pb
    output_pb = args.out_pb
    input_node_names = args.input_layers.split(",")
    output_node_names = args.output_layers.split(",")
    transforms = args.transforms.split()

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(input_pb, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=graph) as sess:
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), input_node_names, output_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, output_node_names)

        head, tail = os.path.split(output_pb)
        graph_io.write_graph(constant_graph, head, tail, as_text=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_pb', type=str, required=True)
    parser.add_argument('--out_pb', type=str, required=True)
    parser.add_argument('--input_layers', type=str, required=True)
    parser.add_argument('--output_layers', type=str, required=True)
    parser.add_argument('--transforms', type=str, default="""
            strip_unused_nodes(type=float)
            remove_nodes(op=Identity, op=CheckNumerics)
            fold_constants(ignore_errors=true)
            fold_batch_norms
            fold_old_batch_norms
    """)
    args = parser.parse_args()
    main(args)
