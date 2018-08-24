ARG tf_ver=1.8.0
FROM tensorflow:${tf_ver}

# Build Tensorflow C++ Library [Only test on Tensroflow 1.5]
# Workaround to build Tensorflow with --config=monolithic. Otherwise, Opencv cannot read jpeg.
# Follow the issue on https://github.com/tensorflow/tensorflow/issues/1924
RUN cd /tensorflow \
    && printf '\n\n\n\n\n\n\n\n\n\n\n\n' | ./configure \
    && bazel build --config=monolithic //tensorflow:libtensorflow_cc.so \
    && mkdir -p                /usr/local/include/google/tensorflow \
    && cp -r bazel-genfiles/*  /usr/local/include/google/tensorflow/ \
    && cp -r tensorflow        /usr/local/include/google/tensorflow/ \
    && find                    /usr/local/include/google/tensorflow -type f  ! -name "*.h" -delete \
    && cp -r third_party       /usr/local/include/google/tensorflow/ \
    && cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib

RUN cd /tensorflow \
    && tensorflow/contrib/makefile/download_dependencies.sh

# Build protobuf (Tensorflow dependency)
RUN cd /tensorflow/tensorflow/contrib/makefile/downloads/protobuf \
    && ./autogen.sh \
    && ./configure \
    && make install

# Build Eigen (Tensorflow dependency)
RUN cd /tensorflow/tensorflow/contrib/makefile/downloads/eigen \
    && mkdir build && cd build \
    && cmake .. \
    && make install

# Build Nsync (Tensorflow dependency)
RUN git clone https://github.com/google/nsync.git \
    && cd nsync \
    && git checkout 839fcc53ff9be58218ed55397deb3f8376a1444e \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr \
    && make -j4 install \
    && cd .. \
    && rm -Rf nsync

# [Warning] - Do not use root in public IP 

CMD jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 & \
    tensorboard --logdir=/log/tensorboard