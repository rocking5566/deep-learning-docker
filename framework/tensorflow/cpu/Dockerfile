ARG tf_ver=1.8.0
FROM tensorflow/tensorflow:${tf_ver}-devel-py3

RUN pip3 install pillow h5py

# ============= Tensorflow hub =============
RUN pip3 install tensorflow-hub

# ============= Tensorflow object detection API =============
RUN apt update && \
    apt install -y \
      wget

RUN cd /tensorflow \
    && tensorflow/contrib/makefile/download_dependencies.sh

RUN apt-get install -y \
      autoconf \
      libtool

# Build protobuf3 (Tensorflow object detection dependency)
RUN cd /tensorflow/tensorflow/contrib/makefile/downloads/protobuf \
    && ./autogen.sh \
    && ./configure \
    && make install
RUN ldconfig

ARG tf_model_ver=32e7d660
RUN git clone https://github.com/tensorflow/models.git /opt/tf_model && cd /opt/tf_model && git checkout ${tf_model_ver}
RUN cd /opt/tf_model/research && protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH='$PYTHONPATH:/opt/tf_model/research:/opt/tf_model/research/slim'

# install coco api for making tf_record and evaluating metric
RUN git clone https://github.com/cocodataset/cocoapi.git /opt/cocoapi
RUN pip3 install cython
RUN cd /opt/cocoapi/PythonAPI && sed -i 's/python/python3/g' Makefile && make install

# TF object detection API dependencies
RUN pip3 install contextlib2

# ============= OpenCV3 =============
RUN apt update && \
    apt install -y \
      libsm6 \
      libxext6
RUN pip3 install opencv-python

# ============= CMake & Swig =============
RUN apt update && \
    apt install -y \
      cmake \
      swig

WORKDIR /
