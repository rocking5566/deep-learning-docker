DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=nvidia-docker
PYTORCH_VER?=1.4
IMAGE_NAME=rocking_pytorch:$(PYTORCH_VER)
REPO_ROOT?=$(shell git rev-parse --show-toplevel)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
  X11_DISPLAY=--env DISPLAY=$(DISPLAY) \
  	--env="QT_X11_NO_MITSHM=1" \
  	-v /tmp/.X11-unix:/tmp/.X11-unix:ro
endif
ifeq ($(UNAME_S), Darwin)
  X11_DISPLAY=--env DISPLAY=docker.for.mac.localhost:0 \
  	--env="QT_X11_NO_MITSHM=1" \
  	-v /tmp/.X11-unix:/tmp/.X11-unix:ro
endif

MOUNT_DIR=-v $(REPO_ROOT):/workspace

MOUNT_WEBCAM=--device /dev/video0:/dev/video0

help:
	@echo "'make build' will build docker image with tensorflow" $(TF_VER)
	@echo "For other tensorflow version. For example,"
	@echo "'make build TF_VER=1.4.0' will build docker image with tensorflow 1.4.0"
	@echo "'make x11' will run docker container supported x11"
	@echo "'make notebook' will run docker container for ipython notebook & Tensorboard"

build:
	$(DOCKER) build -t $(IMAGE_NAME) --build-arg pytorch_ver=$(PYTORCH_VER) -f $(DOCKER_FILE) .

bash:
	$(DOCKER) run -it -w /workspace --privileged --rm $(MOUNT_DIR) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash

x11:
	xhost +
	$(DOCKER) run -it -w /workspace --privileged --rm $(MOUNT_DIR) $(X11_DISPLAY) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash
	xhost -

