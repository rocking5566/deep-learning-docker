import os
import tensorflow as tf
from pathlib import Path
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


def save_frozen_tf_model(keras_model, tf_model_name, dst_dir='.',
                         num_output=1, graph_def=False,
                         output_graphdef_name='model.ascii', quantize=False):
    if dst_dir != '.':
        Path(dst_dir).mkdir(parents=True, exist_ok=True)

    tf_model_full_path = os.path.join(dst_dir, tf_model_name)
    if os.path.exists(tf_model_full_path):
        os.remove(tf_model_full_path)

    keras_pred_node_names = [x.op.name for x in keras_model.outputs]
    pred = [None] * num_output
    pred_node_names = [None] * num_output

    for i in range(num_output):
        pred_node_names[i] = keras_pred_node_names[i]
        pred[i] = tf.identity(keras_model.outputs[i], name=pred_node_names[i])

    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    # write graph definition in ascii
    if graph_def:
        graph_def_full_path = os.path.join(dst_dir, output_graphdef_name)
        if os.path.exists(graph_def_full_path):
            os.remove(graph_def_full_path)

        tf.train.write_graph(sess.graph.as_graph_def(), dst_dir, output_graphdef_name, as_text=True)
        print('saved the graph definition in ascii format at: ', graph_def_full_path)

    # convert variables to constants and save
    if quantize:
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)

    graph_io.write_graph(constant_graph, dst_dir, tf_model_name, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', tf_model_full_path)
