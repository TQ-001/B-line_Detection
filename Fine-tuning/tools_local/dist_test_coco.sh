#!/usr/bin/env bash

set -e
set -x

PTH=$1
WORK_DIR=$2
PORT=${PORT:-29500}


cd benchmarks/detection/


# COCO R50-FPN 12k
#bash run.sh configs/coco_R_50_FPN_12k_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_FPN_12k

#python visualize_json_results.py \
#	--input ../../$WORK_DIR/coco_R_50_FPN_12k/inference/coco_instances_results.json \
#	--output ../../$WORK_DIR/coco_R_50_FPN_12k/inference/ \
#	--dataset lus_val

#echo 'coco_R_50_FPN_12k finish!!!'

# COCO RetinaNet 12k
#bash run.sh configs/coco_R_50_RetinaNet_12k_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_RetinaNet_12k
#echo 'coco_retinanet_R_50_FPN_12k finish!!!'
# COCO R50-C4 12k
#bash run.sh configs/coco_R_50_C4_12k_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_C4_12k
#echo 'coco_R_50_C4_12k finish!!!'



# COCO R50-FPN 1x
bash run.sh configs/coco_R_50_FPN_1x_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_FPN_1x

python visualize_json_results.py \
	--input ../../$WORK_DIR/coco_R_50_FPN_1x/inference/coco_instances_results.json \
	--output ../../$WORK_DIR/coco_R_50_FPN_1x/inference/ \
	--dataset lus_val

echo 'coco_R_50_FPN_1x finish!!!'
# COCO RetinaNet 1x
#bash run.sh configs/coco_R_50_RetinaNet_1x_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_RetinaNet_1x

#python visualize_json_results.py \
#	--input ../../$WORK_DIR/coco_R_50_RetinaNet_1x/inference/coco_instances_results.json \
#	--output ../../$WORK_DIR/coco_R_50_RetinaNet_1x/inference/ \
#	--dataset lus_val

#echo 'coco_retinanet_R_50_FPN_1x finish!!!'
# COCO R50-C4 1x
# bash run.sh configs/coco_R_50_C4_1x_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_C4_1x

#  python visualize_json_results.py \
#	--input ../../$WORK_DIR/coco_R_50_C4_1x/inference/coco_instances_results.json \
#	--output ../../$WORK_DIR/coco_R_50_C4_1x/inference/ \
#	--dataset lus_val

# echo 'coco_R_50_C4_1x finish!!!'





# COCO R50-FPN 2x
#bash run.sh configs/coco_R_50_FPN_2x_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_FPN_2x

#python visualize_json_results.py \
#	--input ../../$WORK_DIR/coco_R_50_FPN_2x/inference/coco_instances_results.json \
#	--output ../../$WORK_DIR/coco_R_50_FPN_2x/inference/ \
#	--dataset lus_val

#echo 'coco_R_50_FPN_2x finish!!!'
# COCO RetinaNet 2x
# bash run.sh configs/coco_R_50_RetinaNet_2x_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_RetinaNet_2x
# echo 'coco_retinanet_R_50_FPN_2x finish!!!'
# COCO R50-C4 2x
# bash run.sh configs/coco_R_50_C4_2x_moco.yaml ../../$PTH ../../$WORK_DIR/coco_R_50_C4_2x
# echo 'coco_R_50_C4_2x finish!!!'


cd -