#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

# Arguments
MODEL="meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER="meta-llama/Llama-3.2-1B-Instruct"
TASKS="arc_challenge,hellaswag,winogrande"
DEVICE="cuda"

# Parse command-line options
while getopts m:t:d:f: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        t) TOKENIZER=${OPTARG};;
        d) DEVICE=${OPTARG};;
        f) NUM_FEWSHOT=${OPTARG};;
        *) echo "Invalid option";;
    esac
done

# Run the script
python ${SCRIPT_DIR}/"eval_v1.py" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --num_fewshot ${NUM_FEWSHOT}
