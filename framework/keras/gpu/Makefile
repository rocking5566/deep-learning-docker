DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=nvidia-docker
TF_VER?=1.8.0
IMAGE_NAME=keras:tf_$(TF_VER)
NCNN_IMAGE_NAME=keras:ncnn
REPO_ROOT?=$(shell dirname `pwd`)/../../

X11_DISPLAY=--env DISPLAY=$(DISPLAY) \
	--env="QT_X11_NO_MITSHM=1" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:ro

MOUNT_DIR=-v $(REPO_ROOT):/workspace \
	-v $(DATA):/datasets

MOUNT_WEBCAM=--device /dev/video0:/dev/video0

help:
	@echo "'make build' will build docker keras image with backend - tensorflow" $(TF_VER)
	@echo "For other tensorflow version. For example,"
	@echo "'make build TF_VER=1.4.0' will build docker keras image with backend - tensorflow 1.4.0"
	@echo "'make x11' will run docker container supported x11"
	@echo "'make notebook' will run docker container for ipython notebook & Tensorboard"

build_all: build build_ncnn
build:
	cd ../../tensorflow/tf_object_detection && make build TF_VER=$(TF_VER)
	$(DOCKER) build -t $(IMAGE_NAME) --build-arg tf_ver=$(TF_VER) -f $(DOCKER_FILE) .
build_ncnn:
	$(DOCKER) build -t $(NCNN_IMAGE_NAME) --build-arg tf_ver=$(TF_VER) -f Dockerfile.ncnn .

bash:
	$(DOCKER) run -it --privileged --rm $(MOUNT_DIR) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash

x11:
	xhost +
	$(DOCKER) run -it --privileged --rm $(MOUNT_DIR) $(X11_DISPLAY) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash
	xhost -

ipython:
	$(DOCKER) run -it --rm $(MOUNT_DIR) $(IMAGE_NAME) ipython

notebook:
	$(DOCKER) run -it --rm $(MOUNT_DIR) --net=host $(IMAGE_NAME)

ncnn:
	$(DOCKER) run -it --privileged --rm $(MOUNT_DIR) $(X11_DISPLAY) $(MOUNT_WEBCAM) --net=host $(NCNN_IMAGE_NAME) bash
