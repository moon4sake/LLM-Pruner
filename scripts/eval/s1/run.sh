#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors and output error message
trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

folder_path='eval/s1'
models=(
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
)
architecture_names=("qwen")
sparsity_values=("0.10")

run_pipeline() {
    local base_model=$1
    local architecture_name=$2
    local sparsity=$3
    local gpu_id=$4
    local name=${base_model##*/}
    prune_ckpt_path="prune_log/${name}_s${sparsity}_block_all_global/pytorch_model.bin"
    tune_ckpt_path="tune_log/${name}_s${sparsity}_block_all_global/checkpoint-5000"
    out_path="results/s1/${name}_s${sparsity}_block_all_global"

    echo "[${name} - Sparsity: ${sparsity}] [START] - Start Evaluation on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python ${folder_path}/s1_score_transformers.py \
        --model ${prune_ckpt_path} \
        --adapter ${tune_ckpt_path} \
        --architecture_name ${architecture_name} \
        --data_file ${folder_path}/data/gsm8k_test.csv \
        --outdir ${out_path} \
        --tensor_parallel_size 4
    echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Evaluation"
}

# Set the GPUs to use
gpus=(7)

# Main loop to run each model
for model in "${models[@]}"; do
    echo "Starting processing for model: ${model}"
    
    for j in "${!sparsity_values[@]}"; do
        architecture_name="${architecture_names[$j]}"
        sparsity="${sparsity_values[$j]}"
        gpu_id="${gpus[$((j % ${#gpus[@]}))]}"
        (
            run_pipeline "$model" "$architecture_name" "$sparsity" "$gpu_id"
        ) &
    done
    
    wait  # Wait for all sparsity jobs for the current model to finish
    echo "Completed processing for model: ${model}"
done

echo "All processing completed."