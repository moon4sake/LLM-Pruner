#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

# Arguments
MODEL="meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER="meta-llama/Llama-3.2-1B-Instruct"
TASKS="arc_challenge,arc_easy,hellaswag,winogrande"
DEVICE="cuda"

# Parse command-line options
while getopts m:t:d: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;
        t) TOKENIZER=${OPTARG};;
        d) DEVICE=${OPTARG};;
        *) echo "Invalid option";;
    esac
done

# Run the script
python ${SCRIPT_DIR}/"eval_sft_pruned.py" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}"
