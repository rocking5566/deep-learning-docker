DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=nvidia-docker
TF_VER?=1.13.1
IMAGE_NAME=rocking_tensorflow:$(TF_VER)
REPO_ROOT?=$(shell git rev-parse --show-toplevel)
CONTAINER_NAME="tensorflow_docker_test"
TMP_DIR=/tmp

X11_DISPLAY=--env DISPLAY=$(DISPLAY) \
	--env="QT_X11_NO_MITSHM=1" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:ro

MOUNT_DIR=-v $(REPO_ROOT):/workspace \
	-v $(DATA):/datasets \
	-v $(TMP_DIR):/tmp

MOUNT_WEBCAM=--device /dev/video0:/dev/video0

help:
	@echo "'make build' will build docker image with tensorflow" $(TF_VER)
	@echo "For other tensorflow version. For example,"
	@echo "'make build TF_VER=1.4.0' will build docker image with tensorflow 1.4.0"
	@echo "'make x11' will run docker container supported x11"
	@echo "'make notebook' will run docker container for ipython notebook & Tensorboard"

build:
	$(DOCKER) build -t $(IMAGE_NAME) --build-arg tf_ver=$(TF_VER)-gpu-py3-jupyter -f $(DOCKER_FILE) .

bash:
	$(DOCKER) run -it -w /workspace --privileged -u $(shell id -u):$(shell id -g) --rm $(MOUNT_DIR) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash

x11:
	xhost +
	$(DOCKER) run -it -w /workspace --privileged -u $(shell id -u):$(shell id -g) --rm $(MOUNT_DIR) $(X11_DISPLAY) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash
	xhost -

notebook:
	$(DOCKER) run -it -w /workspace --privileged -u $(shell id -u):$(shell id -g) --rm $(MOUNT_DIR) --net=host $(IMAGE_NAME)

dev:
ifeq ($(shell docker ps -aqf name=${CONTAINER_NAME}),)
	$(DOCKER) run -it -w /workspace --privileged -u $(shell id -u):$(shell id -g) --name ${CONTAINER_NAME} $(MOUNT_DIR) --net=host $(IMAGE_NAME) bash
else
	$(DOCKER) start -i ${CONTAINER_NAME}
endif
