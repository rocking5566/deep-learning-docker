# SSD: Single Shot MultiBox Detector

In order to train an SSD300 or SSD512 from scratch, download the weights of the fully convolutionalized VGG-16 model trained to convergence on ImageNet classification here:
[vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5](https://drive.google.com/open?id=0B0WbA4IemlxlbFZZaURkMTl2NVU)

This is a modified version of the VGG-16 model from keras.applications.vgg16. In particular, the fc6 and fc7 layers were convolutionalized and sub-sampled from depth 4096 to 1024, following the paper.

## Dependencies
Python 3.x
Numpy
TensorFlow 1.x
Keras 2.x
OpenCV (for data augmentation)
Beautiful Soup 4.x (to parse XML files)
