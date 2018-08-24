ARG tf_ver=1.8.0
FROM tensorflow:${tf_ver}

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

# install coco api for making tf_record and evaluating metric
RUN git clone https://github.com/cocodataset/cocoapi.git /opt/cocoapi
RUN cd /opt/cocoapi/PythonAPI && sed -i 's/python/python3/g' Makefile && make install

# TF object detection API dependencies
RUN pip3 install contextlib2
