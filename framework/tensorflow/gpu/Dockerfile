ARG tf_ver=2.1.0-gpu-py3-jupyter
FROM tensorflow/tensorflow:${tf_ver}

ENV DEBIAN_FRONTEND=noninteractive

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
    libswscale-dev \
    python3-tk \
    autoconf \
    automake \
    libtool \
    swig \
    wget \
    git \
    zip \
    zlib1g-dev

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
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3.7) \
    -DPYTHON_INCLUDE_DIR=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3.6 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    .. \
  && make -j"$(nproc)" && make install \
  && cd /opt \
  && rm ${OPENCV_VERSION}.zip \
  && rm -r opencv-${OPENCV_VERSION}

RUN pip install git+https://github.com/microsoft/onnxconverter-common && \
    pip install git+https://github.com/onnx/keras-onnx

WORKDIR /
CMD jupyter notebook --allow-root --port=8888 & \
    tensorboard --logdir=/workspace/tensorboard
