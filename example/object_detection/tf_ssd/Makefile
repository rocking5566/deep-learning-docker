MODEL=model_finetune_1.0-MobileNetV1-300

# freeze settings
STEP_TO_FREEZE=0

tfrecord:
	cd /opt/tf_model/research && \
	python3 object_detection/dataset_tools/create_coco_tf_record.py \
	    --logtostderr \
	    --train_image_dir="/datasets/coco_2017/train2017" \
	    --val_image_dir="/datasets/coco_2017/val2017" \
	    --test_image_dir="/datasets/coco_2017/test2017" \
	    --train_annotations_file="/datasets/coco_2017/annotations/instances_train2017.json" \
	    --val_annotations_file="/datasets/coco_2017/annotations/instances_val2017.json" \
	    --testdev_annotations_file="/datasets/coco_2017/annotations/image_info_test-dev2017.json" \
	    --output_dir="/datasets/coco_2017"
train:
	cd /opt/tf_model/research && \
	python3 object_detection/train.py \
	    --logtostderr \
	    --pipeline_config_path=/workspace/example/object_detection/tf_ssd/models/${MODEL}/config \
	    --train_dir=/workspace/example/object_detection/tf_ssd/models/${MODEL}/train
eval:
	cd /opt/tf_model/research && \
	CUDA_VISIBLE_DEVICES="" \
	python3 object_detection/eval.py \
	    --logtostderr \
	    --pipeline_config_path=/workspace/example/object_detection/tf_ssd/models/${MODEL}/config \
	    --checkpoint_dir=/workspace/example/object_detection/tf_ssd/models/${MODEL}/train \
	    --eval_dir=/workspace/example/object_detection/tf_ssd/models/${MODEL}/eval
tb:
	tensorboard --logdir=/workspace/example/object_detection/tf_ssd/models/${MODEL}
freeze:
	cd /opt/tf_model/research && \
	CUDA_VISIBLE_DEVICES="" \
	python3 object_detection/export_inference_graph.py \
	    --input_type image_tensor \
	    --pipeline_config_path=/workspace/example/object_detection/tf_ssd/models/${MODEL}/config \
	    --trained_checkpoint_prefix=/workspace/example/object_detection/tf_ssd/models/${MODEL}/train/model.ckpt-${STEP_TO_FREEZE} \
	    --output_directory=/workspace/example/object_detection/tf_ssd/models/${MODEL}/freeze/ \
	    --config_override " \
	        model{ \
	          ssd { \
	            post_processing { \
	              batch_non_max_suppression { \
	                score_threshold: 1e-8 \
	              } \
	            } \
	          } \
	        }"
demo:
	cd /workspace/tool && \
	CUDA_VISIBLE_DEVICES="" \
	python3 tf_objdet_api_webcam.py \
	    --pbtxt /opt/tf_model/research/object_detection/data/mscoco_label_map.pbtxt \
	    --pb /workspace/example/object_detection/tf_ssd/models/${MODEL}/freeze/frozen_inference_graph.pb \
	    --num_classes 90 \
	    --min_score_thresh 0.3
# NOTE: run this command in Tensorflow SNPE docker container
dlc: SHELL:=/bin/bash
dlc:
	cd /opt/snpe && \
	source bin/envsetup.sh -t /tensorflow && \
	CUDA_VISIBLE_DEVICES="" \
	snpe-tensorflow-to-dlc \
	    --graph /workspace/example/object_detection/tf_ssd/models/${MODEL}/freeze/frozen_inference_graph.pb \
	    -i Preprocessor/sub 300,300,3 \
	    --out_node detection_classes \
	    --out_node detection_boxes \
	    --out_node detection_scores \
	    --dlc /workspace/example/object_detection/tf_ssd/models/${MODEL}/freeze/frozen_inference_graph.dlc \
	    --allow_unconsumed_nodes

