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

run_pipeline() {
    local base_model=$1
    local gpu_id=$2
    local name=${base_model##*/}
    out_path="results/gsm8k/${name}"

    echo "[${name}] [START] - Start Evaluation on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python ${folder_path}/eval_transformers.py \
                                            --model ${base_model} \
                                            --tokenizer ${base_model} \
                                            --data_file ${folder_path}/data/test/GSM8K_test.jsonl \
                                            --outdir ${out_path} \
                                            --is_pretrained
    echo "[${name}] [FINISH] - Finish Evaluation"
}

# Set the GPUs to use
gpus=(7)
# Main loop to run each model
for j in "${!models[@]}"; do
    model="${models[$j]}"
    echo "Starting processing for model: ${model}"
    
    gpu_id="${gpus[$((j % ${#gpus[@]}))]}"
    (
        run_pipeline "$model" "$gpu_id"
    ) &
done

# Wait for all model evaluations to complete
wait
echo "All processing completed."