ARG pytorch_ver=1.4
FROM pytorch/pytorch:${pytorch_ver}-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y cmake \
    g++ \
    unzip \
    vim \
    wget \
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

ENV OPENCV_DIR /opt

RUN mkdir -p $OPENCV_DIR 

WORKDIR $OPENCV_DIR

ENV OPENCV_VERSION="4.1.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
  && unzip ${OPENCV_VERSION}.zip \
  && mkdir opencv-${OPENCV_VERSION}/build \
  && cd opencv-${OPENCV_VERSION}/build \
  && cmake -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3.7) \
    -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    .. \
  && make -j"$(nproc)" && make install \
  && cd /opt \
  && rm ${OPENCV_VERSION}.zip \
  && rm -r opencv-${OPENCV_VERSION}

RUN pip install jupyter matplotlib
WORKDIR /