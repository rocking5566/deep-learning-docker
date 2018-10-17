DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=docker
TF_VER?=1.8.0
SNPE_VER?=1.17.0
IMAGE_NAME=tensorflow-snpe:$(TF_VER)-$(SNPE_VER)
REPO_ROOT?=$(shell git rev-parse --show-toplevel)
TMP_DIR=/tmp

MOUNT_DIR=-v $(REPO_ROOT):/workspace \
	-v $(DATA):/datasets \
	-v $(TMP_DIR):/tmp

#MOUNT_USER=-u $(shell id -u):$(shell id -g)

help:
	@echo "'make build' will build docker image with tensorflow" $(TF_VER)
	@echo "For other tensorflow version. For example,"
	@echo "'make build TF_VER=1.4.0' will build docker image with tensorflow 1.4.0"
	@echo "'make x11' will run docker container supported x11"
	@echo "'make notebook' will run docker container for ipython notebook & Tensorboard"

build:
	$(DOCKER) build -t $(IMAGE_NAME) --build-arg tf_ver=$(TF_VER) --build-arg snpe_ver=$(SNPE_VER) -f $(DOCKER_FILE) .

bash:
	$(DOCKER) run -it --privileged --rm $(MOUNT_DIR) --net=host $(IMAGE_NAME) bash
