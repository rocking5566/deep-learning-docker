# Install
From a brand new server, you just need to install docker, nvidia driver, and nvidia-docker.  
You could refer to install scripts in root folder.

# How to start
You should go ./framework/tensorflow/tf_object_detection/, and 
```sh
make build
```
if this is your first time to play with this example, then, simply
```sh
make bash
```
Now, you should enter a environment include all dependencies of TF object detection API.  
The following steps are assuming you are in the docker container, and change to working space(/workspace/example/object_detection/tf_ssd/)

## make tfrecord of dataset
Assuming we have downloaded the coco datasets,  
create a folder(/datasets/coco_2017) which you could put coco dataset in, then
```sh
make tfrecord
```
You should get tfrecords(train, val, test) in the coco folder(/datasets/coco_2017)

## train
vim Makefile and modify MODEL to you would like to use, then just 
```sh
make train
```
## eval
```sh
make eval
```
> Note: You should run this simultaneously with training

In order to execute without opening a new container, you could
```sh
docker ps
```
to check what the container ID is, and then
```sh
docker exec -it CONTAINER_ID bash
```
to enter into the same container which you launched in "train" step.
## monitor process
```sh
make tb # which means tensorboard
```


