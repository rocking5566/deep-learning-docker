'''
Modified from keras SSD300 network.

Copyright (C) 2018 Rocking Lai
'''

import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, BatchNormalization
from keras.regularizers import l2

from keras_layer_AnchorBoxes import AnchorBoxes
#from keras.applications.mobilenet import MobileNet
from network import mobilenetv1

MOBILENET_ALPHA = 0.25


from network.mobilenetv1 import relu6
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), kernel_regularizer=None, block_id=1):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = inputs
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               kernel_regularizer=kernel_regularizer,
               name='conv%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv%s_bn' % block_id)(x)
    return Activation(relu6, name='conv%s_relu' % block_id)(x)


def ssd_300_mobilenet_v1(image_size,
            n_classes,
            l2_regularization=0.0005,
            min_scale=0.1,
            max_scale=0.9,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=None,
            offsets=None,
            limit_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=False,
            subtract_mean=None,
            divide_by_stddev=None,
            swap_channels=True,
            return_predictor_sizes=False):
    '''
    Build a Keras model with SSD_300 architecture, see references.

    The base network is MobileNet, extended by the SSD architecture,
    as described in the paper.

    In case you're wondering why this function has so many arguments: All arguments except
    the first two (`image_size` and `n_classes`) are only needed so that the anchor box
    layers can produce the correct anchor boxes. In case you're training the network, the
    parameters passed here must be the same as the ones used to set up `SSDBoxEncoder`.
    In case you're loading trained weights, the parameters passed here must be the same
    as the ones used to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used.
            Defaults to `None`. If a list is passed, this argument overrides `min_scale` and
            `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers. Defaults to None.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
            Defaults to the aspect ratios used in the original SSD300 architecture, i.e.:
                [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor. Defaults to `True`, following the original
            implementation.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
            This would normally be set to `True`, but here it defaults to `False`, following the original
            implementation.
        variances (list, optional): A list of 4 floats >0 with scaling factors (actually it's not factors but divisors
            to be precise) for the encoded predicted box coordinates. A variance value of 1.0 would apply
            no scaling at all to the predictions, while values in (0,1) upscale the encoded predictions and values greater
            than 1.0 downscale the encoded predictions. Defaults to `[0.1, 0.1, 0.2, 0.2]`, following the original implementation.
            The coordinate format must be 'centroids'.
        coords (str, optional): The box coordinate format to be used. Can be either 'centroids' for the format
            `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids', following the original implementation.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates. Defaults to `False`.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (bool, optional): If `True`, the color channel order of the input images will be reversed,
            i.e. if the input color channel order is RGB, the color channels will be swapped to BGR.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
        https://github.com/chuanqi305/MobileNet-SSD
    '''

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300

    n_classes += 1 # Account for the background class.

    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    l2_reg = l2_regularization

    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ### Build the actual network.

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that subsequent lambda layers can be optional.
    x1 = Lambda(lambda z: z,
                output_shape=(img_height, img_width, img_channels),
                name='idendity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(lambda z: z - np.array(subtract_mean),
                   output_shape=(img_height, img_width, img_channels),
                   name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(lambda z: z / np.array(divide_by_stddev),
                   output_shape=(img_height, img_width, img_channels),
                   name='input_stddev_normalization')(x1)
    if swap_channels and (img_channels == 3):
        x1 = Lambda(lambda z: z[...,::-1],
                   output_shape=(img_height, img_width, img_channels),
                   name='input_channel_swap')(x1)

    # NOTE: dropout(default: 1e-3) here is not important, since dropout is only applied on
    # top(classification) layer, however, we ignore the top layer
    #base_model = keras.applications.mobilenet.MobileNet(input_tensor=x1, weights=None, alpha=MOBILENET_ALPHA, include_top=False)
    base_model = mobilenetv1.MobileNet(input_tensor=x1, weights=None, alpha=MOBILENET_ALPHA, include_top=False, kernel_regularizer=l2(l2_reg))
    conv11 = base_model.get_layer(name='conv_pw_11_relu').output
    conv13 = base_model.get_layer(name='conv_pw_13_relu').output

    def feature_map(input_layer, filters, alpha=1, kernel_regularizer=None, block_id=None):
        middle_layer = _conv_block(input_layer, filters/2, alpha, kernel=(1, 1), strides=(1, 1), kernel_regularizer=kernel_regularizer, block_id='%d_1' % block_id)
        output_layer = _conv_block(middle_layer, filters, alpha, kernel=(3, 3), strides=(2, 2), kernel_regularizer=kernel_regularizer, block_id='%d_2' % block_id)
        return output_layer

    conv14 = feature_map(conv13, 512, MOBILENET_ALPHA, kernel_regularizer=l2(l2_reg), block_id=14)
    conv15 = feature_map(conv14, 256, MOBILENET_ALPHA, kernel_regularizer=l2(l2_reg), block_id=15)
    conv16 = feature_map(conv15, 256, MOBILENET_ALPHA, kernel_regularizer=l2(l2_reg), block_id=16)
    conv17 = feature_map(conv16, 128, MOBILENET_ALPHA, kernel_regularizer=l2(l2_reg), block_id=17)

    ### Build the convolutional predictor layers on top of the base network

    # Classification
    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv11_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same',
                                    kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_mbox_conf')(conv11)
    conv13_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv13_mbox_conf')(conv13)
    conv14_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv14_mbox_conf')(conv14)
    conv15_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv15_mbox_conf')(conv15)
    conv16_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv16_mbox_conf')(conv16)
    conv17_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same',
                               kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv17_mbox_conf')(conv17)

    # Bounding Box
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv11_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same',
                                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_mbox_loc')(conv11)
    conv13_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same',
                          kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv13_mbox_loc')(conv13)
    conv14_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv14_mbox_loc')(conv14)
    conv15_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv15_mbox_loc')(conv15)
    conv16_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv16_mbox_loc')(conv16)
    conv17_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same',
                              kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv17_mbox_loc')(conv17)

    # Anchor boxes (Absolute pixel)
    # Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv11_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], limit_boxes=limit_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords,
                                       name='conv11_mbox_priorbox')(conv11_mbox_loc)

    conv13_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                       two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], limit_boxes=limit_boxes,
                                       variances=variances, coords=coords, normalize_coords=normalize_coords,
                                       name='conv13_mbox_priorbox')(conv13_mbox_loc)

    conv14_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], limit_boxes=limit_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv14_mbox_priorbox')(conv14_mbox_loc)

    conv15_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], limit_boxes=limit_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv15_mbox_priorbox')(conv15_mbox_loc)

    conv16_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], limit_boxes=limit_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv16_mbox_priorbox')(conv16_mbox_loc)

    conv17_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], limit_boxes=limit_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv17_mbox_priorbox')(conv17_mbox_loc)


    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv11_mbox_conf_reshape = Reshape((-1, n_classes), name='conv11_mbox_conf_reshape')(conv11_mbox_conf)
    conv13_mbox_conf_reshape = Reshape((-1, n_classes), name='conv13_mbox_conf_reshape')(conv13_mbox_conf)
    conv14_mbox_conf_reshape = Reshape((-1, n_classes), name='conv14_mbox_conf_reshape')(conv14_mbox_conf)
    conv15_mbox_conf_reshape = Reshape((-1, n_classes), name='conv15_mbox_conf_reshape')(conv15_mbox_conf)
    conv16_mbox_conf_reshape = Reshape((-1, n_classes), name='conv16_mbox_conf_reshape')(conv16_mbox_conf)
    conv17_mbox_conf_reshape = Reshape((-1, n_classes), name='conv17_mbox_conf_reshape')(conv17_mbox_conf)

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv11_mbox_loc_reshape = Reshape((-1, 4), name='conv11_mbox_loc_reshape')(conv11_mbox_loc)
    conv13_mbox_loc_reshape = Reshape((-1, 4), name='conv13_mbox_loc_reshape')(conv13_mbox_loc)
    conv14_mbox_loc_reshape = Reshape((-1, 4), name='conv14_mbox_loc_reshape')(conv14_mbox_loc)
    conv15_mbox_loc_reshape = Reshape((-1, 4), name='conv15_mbox_loc_reshape')(conv15_mbox_loc)
    conv16_mbox_loc_reshape = Reshape((-1, 4), name='conv16_mbox_loc_reshape')(conv16_mbox_loc)
    conv17_mbox_loc_reshape = Reshape((-1, 4), name='conv17_mbox_loc_reshape')(conv17_mbox_loc)

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv11_mbox_priorbox_reshape = Reshape((-1, 8), name='conv11_mbox_priorbox_reshape')(conv11_mbox_priorbox)
    conv13_mbox_priorbox_reshape = Reshape((-1, 8), name='conv13_mbox_priorbox_reshape')(conv13_mbox_priorbox)
    conv14_mbox_priorbox_reshape = Reshape((-1, 8), name='conv14_mbox_priorbox_reshape')(conv14_mbox_priorbox)
    conv15_mbox_priorbox_reshape = Reshape((-1, 8), name='conv15_mbox_priorbox_reshape')(conv15_mbox_priorbox)
    conv16_mbox_priorbox_reshape = Reshape((-1, 8), name='conv16_mbox_priorbox_reshape')(conv16_mbox_priorbox)
    conv17_mbox_priorbox_reshape = Reshape((-1, 8), name='conv17_mbox_priorbox_reshape')(conv17_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv11_mbox_conf_reshape,
                                                       conv13_mbox_conf_reshape,
                                                       conv14_mbox_conf_reshape,
                                                       conv15_mbox_conf_reshape,
                                                       conv16_mbox_conf_reshape,
                                                       conv17_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv11_mbox_loc_reshape,
                                                     conv13_mbox_loc_reshape,
                                                     conv14_mbox_loc_reshape,
                                                     conv15_mbox_loc_reshape,
                                                     conv16_mbox_loc_reshape,
                                                     conv17_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv11_mbox_priorbox_reshape,
                                                               conv13_mbox_priorbox_reshape,
                                                               conv14_mbox_priorbox_reshape,
                                                               conv15_mbox_priorbox_reshape,
                                                               conv16_mbox_priorbox_reshape,
                                                               conv17_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
        # be able to generate the default boxes for the matching process outside of the model during training.
        # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
        # Instead, we'll do it in the batch generator function.
        # The spatial dimensions are the same for the confidence and localization predictors, so we just take those of the conf layers.
        predictor_sizes = np.array([conv11_mbox_conf._keras_shape[1:3],
                                    conv13_mbox_conf._keras_shape[1:3],
                                    conv14_mbox_conf._keras_shape[1:3],
                                    conv15_mbox_conf._keras_shape[1:3],
                                    conv16_mbox_conf._keras_shape[1:3],
                                    conv17_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
