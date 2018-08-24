# category_name="other"
# category_name="lie"
# category_name="sit"
category_name="stand"

anno:
	CUDA_VISIBLE_DEVICES="" python3 annotation.py \
		--annotation_filepath "/datasets/sbr/posture_recognition/annotations/annotation.json" \
		--image_dir "/datasets/sbr/posture_recognition/images" \
		--category_name ${category_name} \
		--detection_pbtxt "/opt/tf_model/research/object_detection/data/mscoco_label_map.pbtxt" \
		--detection_pb "/datasets/models/tf/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb" \
		--detection_num_classes 1 \
		--detection_min_score_thresh 0.6


check:
	python3 ../util/check_annotation.py --image_dir /datasets/sbr/posture_recognition/images --annotation_path /datasets/sbr/posture_recognition/annotations/annotation.json


demo:
	CUDA_VISIBLE_DEVICES="" python3 demo.py \
		--classification_pb "/workspace/project/sbr/posture_recognition/models/ssd300_0.25mobilenetv1/posture_recog.pb" \
		--detection_pbtxt "/workspace/project/sbr/posture_recognition/data/label_map.pbtxt" \
		--detection_pb "/workspace/project/sbr/posture_recognition/models/ssd300_0.25mobilenetv1/freeze/frozen_inference_graph.pb" \
		--detection_num_classes 1 \
		--detection_min_score_thresh 0.5 \
		--detection_max_num_bbox 5


split:
	python3 ../object_detection/split_sbr_dataset.py \
	    --ratio_train_val 0.9 \
	    --annotation_file "/datasets/sbr/posture_recognition/annotations/annotation.json"
