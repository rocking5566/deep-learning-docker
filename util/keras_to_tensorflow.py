import os
import tensorflow as tf
from pathlib import Path
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


def save_frozen_tf_model(keras_model, tf_model_name, dst_dir='.', graph_def=False,
                         output_graphdef_name='model.ascii', quantize=False):
    if dst_dir != '.':
        Path(dst_dir).mkdir(parents=True, exist_ok=True)

    tf_model_full_path = os.path.join(dst_dir, tf_model_name)
    if os.path.exists(tf_model_full_path):
        os.remove(tf_model_full_path)

    keras_pred_node_names = [x.op.name for x in keras_model.outputs]
    print('output nodes names are: ', keras_pred_node_names)

    sess = K.get_session()
    # get graph definition
    gd = sess.graph.as_graph_def()

    # fix bug. Ref: https://github.com/tensorflow/tensorflow/issues/3628
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # write graph definition in ascii
    if graph_def:
        graph_def_full_path = os.path.join(dst_dir, output_graphdef_name)
        if os.path.exists(graph_def_full_path):
            os.remove(graph_def_full_path)

        tf.train.write_graph(gd, dst_dir, output_graphdef_name, as_text=True)
        print('saved the graph definition in ascii format at: ', graph_def_full_path)

    # convert variables to constants and save
    constant_graph = graph_util.convert_variables_to_constants(sess, gd, keras_pred_node_names)

    graph_io.write_graph(constant_graph, dst_dir, tf_model_name, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', tf_model_full_path)
