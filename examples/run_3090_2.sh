#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors and output error message
trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

RECIPE_DIR=$(dirname $(dirname $(realpath "$0")))/"models"/"configs"
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"
EXAMPLE_DIR=$(dirname $(dirname $(realpath "$0")))/"examples"

MODELS=(
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
)

SPARSITY_VALUES=("0.10" "0.25" "0.50" "0.75")

# Function to prune, fine-tune, and evaluate a model for a single sparsity level
run_pipeline() {
    local BASE_MODEL=$1
    local SPARSITY=$2
    local GPU_ID=$3
    local NAME=${BASE_MODEL##*/}

    EXP_NAME="${NAME}/${NAME}_s${SPARSITY}_block_all_global"
    DATA_PATH="open-r1/OpenThoughts-114k-math"

    #################
    # Pruning 
    #################
    # echo "[${NAME} - Sparsity: ${SPARSITY} - Iterative Steps: $(( ${SPARSITY/./} / 10 + 1))] [START] - Start Pruning Model on GPU ${GPU_ID}"
    # echo y | CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_DIR}/prune_v1.py --base_model ${BASE_MODEL} \
    #              --pruning_ratio ${SPARSITY} --global_pruning \
    #              --device cpu --eval_device cpu \
    #              --channel_wise --block_wise \
    #              --save_ckpt_log_name ${EXP_NAME} \
    #              --pruner_type taylor --save_model \
    #              --max_seq_len 2048 --iterative_steps $((${SPARSITY/./} / 10 + 1))
    # echo "[${NAME} - Sparsity: ${SPARSITY} - Iterative Steps: $(( ${SPARSITY/./} / 10 + 1))] [FINISH] - Finish Pruning Model"

    #################
    # Fine-tuning
    #################
    echo "[${NAME} - Sparsity: ${SPARSITY}] [START] - Start Tuning on GPU ${GPU_ID}"
    echo y | CUDA_VISIBLE_DEVICES=${GPU_ID} bash ${EXAMPLE_DIR}/train.sh -m ${NAME} -e ${EXP_NAME} -d ${DATA_PATH}
    echo "[${NAME} - Sparsity: ${SPARSITY}] [FINISH] - Finish Prune and Post-Training."

    # #################
    # # Evaluating
    # #################
    # echo "[${NAME} - Sparsity: ${SPARSITY}] [START] - Start Evaluation on GPU ${GPU_ID}"
    # if [ "$SPARSITY" = "0.00" ]; then
    #     echo y | CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/evaluate.sh simple ${BASE_MODEL} ${SPARSITY} "" prune_log/${NAME}_s${SPARSITY}_block 0 0
    # else
    #     echo y | CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/evaluate.sh simple ${BASE_MODEL} ${SPARSITY} tune_log/${TUNED_MODEL_PATH} prune_log/${PRUNED_MODEL_PATH} 2400 0
    # fi
    # echo "[${NAME} - Sparsity: ${SPARSITY}] [FINISH] - Finish Evaluation"
    # echo "[${NAME} - Sparsity: ${SPARSITY}] [INFO] - The pruned model is at ${PRUNED_MODEL_PATH}/pytorch_model.bin, and the recovery weight is at ${TUNED_MODEL_PATH}/"
}

# Set the GPUs to use
# You want to use GPU 2 and 3
gpus=(7)

# Main loop to run each model
for model in "${MODELS[@]}"; do
    echo "Starting processing for model: ${model}"
    
    for i in "${!SPARSITY_VALUES[@]}"; do
        sparsity="${SPARSITY_VALUES[$i]}"
        gpu_id="${gpus[$((i % ${#gpus[@]}))]}"  # Cycle through GPUs 2 and 3
        # Run each sparsity level concurrently for the current model
        (
            run_pipeline "$model" "$sparsity" "$gpu_id"
        )
    done
    
    # Wait for all sparsity jobs for the current model to finish
    wait
    echo "Completed processing for model: ${model}"
done

echo "All processing completed."
