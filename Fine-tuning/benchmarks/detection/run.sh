#!/bin/bash
DET_CFG=$1
WEIGHTS=$2
OUTPUT_DIR=$3

python $(dirname "$0")/train.py --config-file $DET_CFG \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 4 \
    MODEL.WEIGHTS $WEIGHTS \
    OUTPUT_DIR $OUTPUT_DIR 

