#!/bin/sh

# Private variables
RECIPE_DIR=$(dirname $(dirname $(realpath "$0")))/"recipes"
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

# Arguments
MODEL_NAME="Llama-3.2-1B-Instruct"
TASK="sft"
ACCELERATE_STRATEGY="deepspeed_zero3"
NUM_PROCESSES=1

# Parse command-line options
while getopts m:t:p: flag
do
    case "${flag}" in
        m) MODEL_NAME=${OPTARG};;
        t) TASK=${OPTARG};;
        a) ACCELERATE_STRATEGY=${OPTARG};;
        p) NUM_PROCESSES=${OPTARG};;
        *) echo "Invalid option";;
    esac
done

# Run the script
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${RECIPE_DIR}/accelerate_configs/${ACCELERATE_STRATEGY}.yaml \
    --num_processes=${NUM_PROCESSES} \
    ${SCRIPT_DIR}/run_${TASK}_pruned.py ${RECIPE_DIR}/${MODEL_NAME}/${TASK}/config_full.yaml
