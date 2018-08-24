MODEL=ssd300_0.25mobilenetv1

# freeze settings
STEP_TO_FREEZE=0

dataset:
	cd /workspace/tool && \
	python3 extract_coco_dataset.py \
	    --reorder_category_id \
	    --category_names "person" \
	    --src_annotation_path "/datasets/coco_2017/annotations/instances_train2017.json" \
	    --dest_annotation_path "/datasets/sbr/people_detection/annotations/instances_train2017.json"
	cd /workspace/tool && \
	python3 extract_coco_dataset.py \
	    --reorder_category_id \
	    --category_names "person" \
	    --src_annotation_path "/datasets/coco_2017/annotations/instances_val2017.json" \
	    --dest_annotation_path "/datasets/sbr/people_detection/annotations/instances_val2017.json"
# NOTE: testdev_annotations_file here is not used, so just give it a value to
# mock the create_coco_tf_record.py
tfrecord:
	cd /opt/tf_model/research && \
	python3 object_detection/dataset_tools/create_coco_tf_record.py \
	    --logtostderr \
	    --train_image_dir="/datasets/coco_2017/train2017" \
	    --val_image_dir="/datasets/coco_2017/val2017" \
	    --test_image_dir="/datasets/coco_2017/test2017" \
	    --train_annotations_file="/datasets/coco_2017/annotations/instances_train2017.json" \
	    --val_annotations_file="/datasets/sbr/people_detection/annotations/instances_val2017.json" \
	    --testdev_annotations_file="/datasets/coco_2017/annotations/image_info_test-dev2017.json" \
	    --output_dir="/datasets/sbr/people_detection"
train:
	cd /opt/tf_model/research && \
	python3 object_detection/train.py \
	    --logtostderr \
	    --pipeline_config_path=/workspace/project/sbr/people_detection/models/${MODEL}/config \
	    --train_dir=/workspace/project/sbr/people_detection/models/${MODEL}/train
eval:
	cd /opt/tf_model/research && \
	CUDA_VISIBLE_DEVICES="" \
	python3 object_detection/eval.py \
	    --logtostderr \
	    --pipeline_config_path=/workspace/project/sbr/people_detection/models/${MODEL}/config \
	    --checkpoint_dir=/workspace/project/sbr/people_detection/models/${MODEL}/train \
	    --eval_dir=/workspace/project/sbr/people_detection/models/${MODEL}/eval
tb:
	tensorboard --logdir=/workspace/project/sbr/people_detection/models/${MODEL}
freeze:
	cd /opt/tf_model/research && \
	CUDA_VISIBLE_DEVICES="" \
	python3 object_detection/export_inference_graph.py \
	    --input_type image_tensor \
	    --pipeline_config_path=/workspace/project/sbr/people_detection/models/${MODEL}/config \
	    --trained_checkpoint_prefix=/workspace/project/sbr/people_detection/models/${MODEL}/train/model.ckpt-${STEP_TO_FREEZE} \
	    --output_directory=/workspace/project/sbr/people_detection/models/${MODEL}/freeze
demo:
	cd /workspace/tool && \
	CUDA_VISIBLE_DEVICES="" \
	python3 tf_objdet_api_webcam.py \
	    --pbtxt /workspace/project/sbr/people_detection/data/label_map.pbtxt \
	    --pb /workspace/project/sbr/people_detection/models/${MODEL}/freeze/frozen_inference_graph.pb \
	    --num_classes 1
