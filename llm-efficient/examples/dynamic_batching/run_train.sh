#!/bin/sh

# Private variables
RECIPE_DIR=$(dirname $(dirname $(realpath "$0")))/"recipes"
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Arguments
MODEL_NAME="Llama-3.2-1B-Instruct"
TASK="sft"
ACCELERATE_STRATEGY="deepspeed_zero3"
NUM_PROCESSES=1

# Run the script
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${RECIPE_DIR}/accelerate_configs/${ACCELERATE_STRATEGY}.yaml \
    --num_processes=${NUM_PROCESSES} \
    ${SCRIPT_DIR}/dynamic_batching/train_dynamic_batch.py ${RECIPE_DIR}/${MODEL_NAME}/${TASK}/config_full.yaml
