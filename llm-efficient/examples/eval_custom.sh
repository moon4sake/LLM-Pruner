#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"
TASK_DIR=$(dirname $(dirname $(realpath "$0")))/"tasks"

# Environment Setting
set -x  # print the commands

# Arguments
MODEL="Mistral-Nemo-Minitron-2B-128k-Instruct"
TOKENIZER="Mistral-Nemo-Minitron-2B-128k-Instruct"
TASKS="${TASK_DIR}/medbench"
DEVICE="cuda"

# Parse command-line options
while getopts m:t:a:d: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        t) TOKENIZER=${OPTARG};;
        a) TASKS=${OPTARG};;
        d) DEVICE=${OPTARG};;
        *) echo "Invalid option";;
    esac
done

# Run the script
python ${SCRIPT_DIR}/"eval_sft_pruned.py" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --include_path "${TASK_DIR}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}"
    # --output_path "" \
    # --log_samples
