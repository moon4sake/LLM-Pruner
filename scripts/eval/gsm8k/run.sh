#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors and output error message
trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

folder_path='eval/gsm8k'
models=(
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
)
sparsity_values=("0.10")

run_pipeline() {
    local base_model=$1
    local sparsity=$2
    local gpu_id=$3
    local name=${base_model##*/}
    prune_ckpt_path="prune_log/${name}_s${sparsity}_block_all_global/pytorch_model.bin"
    tune_ckpt_path="tune_log/${name}_s${sparsity}_block_all_global/checkpoint-5000"
    out_path="results/gsm8k/${name}_s${sparsity}_block_all_global"

    echo "[${name} - Sparsity: ${sparsity}] [START] - Start Evaluation on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python ${folder_path}/eval_transformers.py \
        --model ${prune_ckpt_path} \
        --adapter ${tune_ckpt_path} \
        --tokenizer ${base_model} \
        --data_file ${folder_path}/data/test/GSM8K_test.jsonl \
        --outdir ${out_path}
    echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Evaluation"
}

# Set the GPUs to use
gpus=(6)

# Main loop to run each model
for model in "${models[@]}"; do
    echo "Starting processing for model: ${model}"
    
    for j in "${!sparsity_values[@]}"; do
        sparsity="${sparsity_values[$j]}"
        gpu_id="${gpus[$((j % ${#gpus[@]}))]}"
        (
            run_pipeline "$model" "$sparsity" "$gpu_id"
        ) &
    done
    
    wait  # Wait for all sparsity jobs for the current model to finish
    echo "Completed processing for model: ${model}"
done

echo "All processing completed."