ARG tf_ver=1.8.0
FROM tensorflow/tensorflow:${tf_ver}-devel

WORKDIR /opt
ARG snpe_ver=1.15.0
# NOTE: official SNPE could not download old version, so that the one way to
# be immutable is to keep one locally.
COPY snpe-${snpe_ver}.zip .
RUN unzip snpe-${snpe_ver}.zip && mv snpe-${snpe_ver} snpe && rm snpe-${snpe_ver}.zip

# these are dependencies snpe needs
RUN apt-get update && apt-get install -y \
  libleveldb-dev \
  libsnappy-dev \
  libopencv-dev \
  libhdf5-serial-dev \
  libboost-all-dev \
  libatlas-base-dev \
  cmake \
  libgflags-dev \
  libgoogle-glog-dev \
  liblmdb-dev \
  libprotobuf-dev \
  protobuf-compiler
RUN apt-get install -y \
  python-dev \
  python-matplotlib \
  python-numpy \
  python-protobuf \
  python-scipy \
  python-skimage \
  python-sphinx \
  python-yaml

WORKDIR /
