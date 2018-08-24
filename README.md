# Docker Environment

## Features
- Three docker images: Tensorflow, Keras, Caffe
- Tensorflow object detection API
- NVIDIA GPU
- Display webcam video in docker container via opencv windows

## Installation
### Ubuntu 16.04

For developer environments
```sh
$ sudo ./install_docker_ce_ubuntu_amd64.sh  # Install docker-ce
$ sudo ./install_nvidia_docker_ubuntu_amd64.sh  # Install nvidia-docker
$ sudo mkdir /root/Data  # Our convention is to put training data here. This folder will be mounted to the docker container.
```

## Create environment in Docker
### Tensorflow
To build the clean Tensorflow 1.8 (GPU) environment
```sh
$ cd {PROJECT_ROOT}/framework/tensorflow/gpu
$ sudo make build   # Build tensorflow DockerFile
```

If you want to build the clean Tensorflow 1.8 (GPU) environment
```sh
$ sudo make build TF_VER=1.8.0
```

To run the clean Tensorflow (GPU) environment
```sh
$ cd {PROJECT_ROOT}/framework/tensorflow/gpu
$ sudo make x11 # or make x11 TF_VER=1.8.0
```

### Keras with Tensorflow backend
To build the clean Keras (GPU) environment
```sh
$ cd {PROJECT_ROOT}/framework/keras/gpu
$ sudo make build   # Build keras docker file
```

To run the clean Keras (GPU) environment
```sh
$ cd {PROJECT_ROOT}/framework/keras/gpu
$ sudo make x11
```

Run the clean Keras (GPU) environment on the web (IPython notebook and TensorBoard)
```sh
$ cd {PROJECT_ROOT}/framework/keras/gpu
$ sudo make notebook
```
You will see the notebook and Tensorboard URL in the terminal

### Caffe
To run the clean caffe environment

```sh
$ cd framework/caffe
$ sudo make bash
```

To Test is caffe environment correct.
Let's train LeNet on MNIST
```sh
$ cd $CAFFE_ROOT
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
$ ./examples/mnist/train_lenet.sh
```

If you see "Optimization Done" in the console like following. Congratulation!

```sh
I1129 03:51:48.452687    39 caffe.cpp:259] Optimization Done.
```