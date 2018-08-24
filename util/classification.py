import numpy as np

# TODO: use inputs, outputs
def classify_objects(imgs_np, sess, graph, inputs=[], outputs=[]):
    image_tensor = graph.get_tensor_by_name('input_cropped_images:0')
    #output = graph.get_tensor_by_name('dense_1/Softmax:0')
    categories = graph.get_tensor_by_name('output_cls_categories/ArgMax:0')
    scores = graph.get_tensor_by_name('output_cls_scores/GatherNd:0')

    (categories, scores) = sess.run(
        [categories, scores],
        feed_dict={image_tensor: imgs_np})
    return categories, scores
