#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors and output error message
trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

# Function to wait for a specific GPU to be free
wait_for_gpu() {
    local gpu_id="$1"

    echo "Checking GPU $gpu_id for availability..."

    while true; do
        # For each process, nvidia-smi prints something like:
        #    GPU,PID,Process name, used memory, ...
        # If there's "No running processes found" for that GPU, or if the only
        # row is the CSV header, the GPU is free.

        # We'll query only compute-apps for that particular GPU:
        # This returns lines for each active process on that GPU.
        processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -v "^$")
        
        # We need the UUID of GPU $gpu_id to filter. Let's get it:
        gpu_uuid=$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader)
        
        # Now filter the 'processes' that match our GPU's uuid
        used_by_this_gpu=$(echo "$processes" | grep "$gpu_uuid" || true)

        if [ -z "$used_by_this_gpu" ]; then
            # If it's empty, means no processes are running on GPU $gpu_id
            echo "GPU $gpu_id is free!"
            break
        else
            echo "GPU $gpu_id is still busy. Checking again in 30s..."
            sleep 30
        fi
    done
}

# Define models and their properties
models=(
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "microsoft/Phi-3-small-128k-instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
)

sparsity_values=("0.00" "0.25" "0.50" "0.75")

# Function to prune, fine-tune, and evaluate a model for a single sparsity level
run_pipeline() {
    local base_model=$1
    local sparsity=$2
    local gpu_id=$3
    local name=${base_model##*/}

    prune_ckpt_path="${name}_s${sparsity}_block_all_global"
    tune_ckpt_path="${name}_s${sparsity}_block_all_global"

    # Evaluating
    echo "[${name} - Sparsity: ${sparsity}] [START] - Start Evaluation on GPU ${gpu_id}"
    if [ "$sparsity" == "0.00" ]; then
        echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate_pro.sh ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_block 0
    else
        echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate_pro.sh ${base_model} ${sparsity} tune_log/${tune_ckpt_path} prune_log/${prune_ckpt_path} 5000
    fi
    echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Evaluation"
    echo "[${name} - Sparsity: ${sparsity}] [INFO] - The pruned model is at prune_log/${prune_ckpt_path}/pytorch_model.bin, and the recovery weight is at tune_log/${tune_ckpt_path}/"
}

# Set the GPUs to use
gpus=(0 1 2 3)

# Main loop to run each model
for i in "${!models[@]}"; do
    echo "Starting processing for model: ${model}"
    model="${models[$i]}"

    for j in "${!sparsity_values[@]}"; do
        sparsity="${sparsity_values[$j]}"
        gpu_id="${gpus[$((j % ${#gpus[@]}))]}"  # Cycle through GPUs 2 and 3
        # Run each sparsity level concurrently for the current model
        (
            run_pipeline "$model" "$sparsity" "$gpu_id"
        ) &
    done
    
    # Wait for all sparsity jobs for the current model to finish
    wait
    echo "Completed processing for model: ${model}"
done
echo "All processing completed."
