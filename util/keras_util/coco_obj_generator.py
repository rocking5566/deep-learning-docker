"""For coco 2017 object detection

"""
import json
import os
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import img_to_array
from numpy.random import uniform
from numpy.random import randint
from PIL import Image as pil_image

# Copy from keras/preprocessing/image.py
# For load_coco_obj_img
_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}

# These methods were only introduced in version 3.4.0 (2016).
if hasattr(pil_image, 'HAMMING'):
    _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
if hasattr(pil_image, 'BOX'):
    _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
# This method is new in version 1.1.3 (2013).
if hasattr(pil_image, 'LANCZOS'):
    _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


class CocoObjGenerator(ImageDataGenerator):
    def __init__(self,
                 bbox_width_shift_range=0.,
                 bbox_height_shift_range=0.,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        self.bbox_width_shift_range = bbox_width_shift_range
        self.bbox_height_shift_range = bbox_height_shift_range
        super(CocoObjGenerator, self).__init__(featurewise_center=featurewise_center,
                                               samplewise_center=samplewise_center,
                                               featurewise_std_normalization=featurewise_std_normalization,
                                               samplewise_std_normalization=samplewise_std_normalization,
                                               rotation_range=rotation_range, width_shift_range=width_shift_range,
                                               height_shift_range=height_shift_range,
                                               fill_mode=fill_mode, cval=cval, horizontal_flip=horizontal_flip,
                                               vertical_flip=vertical_flip, rescale=rescale,
                                               preprocessing_function=preprocessing_function, data_format=data_format)

    def flow_from_directory(self,
                            img_directory, label_file,
                            target_size=(256, 256), color_mode='rgb',
                            batch_size=32, shuffle=True, seed=None,
                            interpolation='nearest'):
        """
        :param img_directory: Path to the image directory
        :param label_file: Path to the target annotation file
        :param target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
        :param color_mode: One of "grayscale", "pseudo_rgb", "rgb", "random". Default: "rgb".
                "pseudo_rgb" means images will convert to grayscale first,
                then copy to another 2 channels to pretend to rgb image.
                "random" means image will be either rgb or pseudo_rgb.
        :param batch_size: Size of the batches of data (default: 32).
        :param shuffle: Whether to shuffle the data (default: True)
        :param seed: Optional random seed for shuffling and transformations.
        :param interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.

        :return:
             A `CocoIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return CocoIterator(self,
                            img_directory, label_file,
                            target_size=target_size, color_mode=color_mode,
                            data_format=self.data_format,
                            batch_size=batch_size, shuffle=shuffle, seed=seed,
                            interpolation=interpolation)

    def load_coco_obj_img(self, path, bbox, color_mode='rgb', target_size=None,
                          interpolation='nearest'):
        """
        :param path:
        :param bbox:
        :param color_mode: One of "grayscale", "pseudo_rgb", "rbg", "random". Default: "rgb".
                "random" means image will be either rgb or "grayscale with 3 channels" (value of 3 channel will be the same)
        :param target_size:
        :param interpolation:
        :return:
        """
        # Loads an image into PIL format.
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if color_mode == "grayscale":
            if img.mode != 'L':
                img = img.convert('L')
        elif color_mode == "rgb":
            if img.mode != 'RGB':
                img = img.convert('RGB')
        elif color_mode == "pseudo_rgb":
            if img.mode != 'L':
                img = img.convert('L')
            img = img.convert('RGB')
        else:
            if (randint(0, 2) < 1) and (img.mode != 'L'):
                img = img.convert('L')

            if img.mode != 'RGB':
                img = img.convert('RGB')

        # Data augmentation - Random shift to crop the object
        tx = uniform(-self.bbox_width_shift_range, self.bbox_width_shift_range)
        ty = uniform(-self.bbox_height_shift_range, self.bbox_height_shift_range)
        x1 = bbox[0] + (tx * bbox[2])
        y1 = bbox[1] + (ty * bbox[3])
        x2 = x1 + bbox[2]
        y2 = y1 + bbox[3]
        img = img.crop([x1, y1, x2, y2])

        # Resize image
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img

class CocoIterator(Iterator):
    def __init__(self, image_data_generator,
                 img_directory, label_file,
                 bbox_width_shift_range=0.,
                 bbox_height_shift_range=0.,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, interpolation='nearest'):
        self.image_data_generator = image_data_generator
        self.img_directory = img_directory
        self.label_file = label_file
        self.bbox_width_shift_range = bbox_width_shift_range
        self.bbox_height_shift_range = bbox_height_shift_range
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'pseudo_rgb', 'grayscale', 'random'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" , "pseudo_rgb", "grayscale", "random".')

        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format

        self.color_mode = color_mode
        if (self.color_mode == 'rgb') or (self.color_mode == 'pseudo_rgb') or (self.color_mode == 'random'):
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.interpolation = interpolation

        self.id_to_img_name = {}
        self.bboxs = []
        self.bbox_to_img_id = []
        self.bbox_to_class_id = []
        self.num_samples = 0
        self.num_classes = 0
        self.parse_label_file()
        super(CocoIterator, self).__init__(self.num_samples, batch_size, shuffle, seed)

    def parse_label_file(self):
        with open(self.label_file, 'r') as fid:
            labels = json.load(fid)
            for img in labels['images']:
                # TODO - Use multiprocessing.pool.ThreadPool().apply_async to speedup parser
                self.id_to_img_name[img['id']] = os.path.join(self.img_directory, img['file_name'])

            # Workaround - categories id is incontinuous in COCO dataset
            # Do not "self.num_classes = len(labels['categories'])"
            for category in labels['categories']:
                if category['id'] > self.num_classes:
                    self.num_classes = category['id']

        for annotation in labels['annotations']:
                bbox = annotation['bbox']
                self.bboxs.append(bbox)
                self.bbox_to_img_id.append(annotation['image_id'])
                self.bbox_to_class_id.append(annotation['category_id'])

        self.num_samples = len(self.bboxs)
        print('Found %d images with %d bbox belonging to %d classes.' % (len(self.id_to_img_name), self.num_samples, self.num_classes))

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array), self.num_classes+1), dtype=K.floatx()) # index 0 for background
        # build batch of image data
        for i, j in enumerate(index_array):
            img_id = self.bbox_to_img_id[j]
            fname = self.id_to_img_name[img_id]
            img = self.image_data_generator.load_coco_obj_img(fname, self.bboxs[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)

            # TODO - data augmentation for cropping the images
            x = img_to_array(img, data_format=self.data_format)

            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            label = self.bbox_to_class_id[j]
            batch_y[i, label] = 1

        return batch_x, batch_y

    def next(self):
        # For python 2.x.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
