FROM caffe2ai/caffe2:c2v0.8.1.cuda8.cudnn7.ubuntu16.04

RUN apt-get update && \
    apt-get install -y cmake \
    g++ \
    unzip \
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