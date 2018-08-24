ARG tf_ver=1.8.0
FROM tensorflow/tensorflow:${tf_ver}-devel-gpu-py3

RUN apt-get update && \
    apt-get install -y cmake \
    libgl1-mesa-glx \
    libhdf5-dev \
    g++ \
    graphviz \
    openmpi-bin \
    vim \
    libgtk2.0-dev \
    pkg-config \
    libcanberra-gtk-module \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# ============= OpenCV =============
# OpenCV
# [Warning] - opencv-python does not include Video I/O (Can not open Webcam)
# Build opencv from source

ARG opencv_version=3.4.1

ENV OPENCV_DIR /opt/opencv

RUN mkdir -p $OPENCV_DIR 

WORKDIR $OPENCV_DIR

RUN wget https://github.com/Itseez/opencv/archive/${opencv_version}.zip -O opencv-${opencv_version}.zip
RUN wget https://github.com/Itseez/opencv_contrib/archive/${opencv_version}.zip -O opencv_contrib-${opencv_version}.zip

RUN unzip opencv-${opencv_version}.zip && \
    unzip opencv_contrib-${opencv_version}.zip && \
    cd opencv-${opencv_version} &&\
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/opencv_contrib-${opencv_version}/modules \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_CUDA=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_OPENCL=OFF \
    -D BUILD_opencv_python3=YES \
  ..

RUN cd opencv-${opencv_version}/build && \
    make -j4 && \
    make install && \
    rm $OPENCV_DIR/opencv-${opencv_version}.zip && \
    rm $OPENCV_DIR/opencv_contrib-${opencv_version}.zip

# ============= tensorpack =============
#RUN pip3 install --upgrade --ignore-installed setuptools && pip3 install --no-cache-dir git+git://github.com/ppwwyyxx/tensorpack

# ============= Tensorflow object detection API =============
RUN apt-get install -y python3-tk autoconf automake libtool

RUN cd /tensorflow \
    && tensorflow/contrib/makefile/download_dependencies.sh

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

# TF object detection API dependencies
RUN pip3 install beautifulsoup4 lxml cython pillow h5py contextlib2

# install coco api for making tf_record and evaluating metric
RUN git clone https://github.com/cocodataset/cocoapi.git /opt/cocoapi
RUN cd /opt/cocoapi/PythonAPI && sed -i 's/python/python3/g' Makefile && make install

# ============= Tensorflow Hub =============
RUN pip3 install tensorflow-hub

# ============= swig =============
RUN apt update && \
    apt install -y \
      swig

WORKDIR /

CMD jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 & \
    tensorboard --logdir=/log/tensorboard
