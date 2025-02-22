#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors and output error message
trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

# Define models and their properties
models=(
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

sparsity_values=("0.25" "0.50")

# Function to prune, fine-tune, and evaluate a model for a single sparsity level
run_pipeline() {
    local base_model=$1
    local sparsity=$2
    local gpu_id=$3
    local name=${base_model##*/}

    prune_ckpt_path="${name}_s${sparsity}_block"
    tune_ckpt_path="${name}_s${sparsity}_block"

    # Pruning with automatic OOM handling
    echo "[${name} - Sparsity: ${sparsity}] [START] - Start Pruning Model on GPU ${gpu_id}"
    if ! (echo y | CUDA_VISIBLE_DEVICES=${gpu_id} python llama3.py --base_model ${base_model} \
        --pruning_ratio ${sparsity} --device cuda --eval_device cuda --block_wise \
        --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 \
        --block_attention_layer_end 30 --save_ckpt_log_name ${prune_ckpt_path} --pruner_type taylor \
        --taylor param_first --max_seq_len 2048 --save_model); then
        
        echo "[${name} - Sparsity: ${sparsity}] [OOM] - OOM error encountered on GPU ${gpu_id}, switching to CPU."
        
        # Retry pruning using CPU if GPU fails with OOM
        CUDA_VISIBLE_DEVICES= python llama3.py --base_model ${base_model} \
            --pruning_ratio ${sparsity} --device cpu --eval_device cuda --block_wise \
            --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 \
            --block_attention_layer_end 30 --save_ckpt_log_name ${prune_ckpt_path} --pruner_type taylor \
            --taylor param_first --max_seq_len 2048 --save_model
    fi
    echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Pruning Model"

    # Fine-tuning
    echo "[${name} - Sparsity: ${sparsity}] [START] - Start Tuning on GPU ${gpu_id}"
    echo y | CUDA_VISIBLE_DEVICES=${gpu_id} python post_training.py --prune_model prune_log/${prune_ckpt_path}/pytorch_model.bin \
        --data_path GAIR/LIMO --output_dir tune_log/${tune_ckpt_path} \
        --wandb_project PruneDebug --lora_r 8 --num_epochs 2 \
        --learning_rate 1e-4 --batch_size 64
    echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Prune and Post-Training."

    # Evaluating
    echo "[${name} - Sparsity: ${sparsity}] [START] - Start Evaluation on GPU ${gpu_id}"
    echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate.sh ${base_model} "" prune_log/${name}_s${sparsity}_block 0
    echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate.sh ${base_model} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
    echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Evaluation"
    echo "[${name} - Sparsity: ${sparsity}] [INFO] - The pruned model is at prune_log/${prune_ckpt_path}/pytorch_model.bin, and the recovery weight is at tune_log/${tune_ckpt_path}/"
}

# Number of available GPUs; adjust if needed
num_gpus=2

# Main loop to run each model
for model in "${models[@]}"; do
    echo "Starting processing for model: ${model}"
    
    for i in "${!sparsity_values[@]}"; do
        sparsity="${sparsity_values[$i]}"
        gpu_id=$((i % num_gpus))  # Cycle through available GPUs
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

# #!/bin/bash

# # Exit immediately if a command exits with a non-zero status
# set -e

# # Function to handle errors and output error message
# trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

# # Define models and their properties
# models=(
#     "meta-llama/Llama-3.1-8B"
#     # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# )

# sparsity_values=("0.10" "0.25" "0.50" "0.75")

# # Function to prune, fine-tune, and evaluate a model for a single sparsity level
# run_pipeline() {
#     local base_model=$1
#     local sparsity=$2
#     local gpu_id=$3
#     local name=${base_model##*/}  # Extract the model's name from path

#     prune_ckpt_path="${name}_s${sparsity}_block"
#     tune_ckpt_path="${name}_s${sparsity}_block"

#     # Pruning
#     echo "[${name} - Sparsity: ${sparsity}] [START] - Start Pruning Model on GPU ${gpu_id}"
#     echo y | CUDA_VISIBLE_DEVICES=${gpu_id} python llama3.py --base_model ${base_model} \
#                                                     --pruning_ratio ${sparsity} \
#                                                     --device cpu --eval_device cuda \
#                                                     --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
#                                                     --block_attention_layer_start 4 --block_attention_layer_end 28 \
#                                                     --save_ckpt_log_name ${prune_ckpt_path} \
#                                                     --pruner_type taylor --taylor param_first \
#                                                     --max_seq_len 2048 \
#                                                     --save_model #--test_before_train --test_after_train 
#     echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Pruning Model"

#     # # Fine-tuning
#     # echo "[${name} - Sparsity: ${sparsity}] [START] - Start Tuning on GPU ${gpu_id}"
#     # echo y | CUDA_VISIBLE_DEVICES=${gpu_id} python post_training.py --prune_model prune_log/${prune_ckpt_path}/pytorch_model.bin \
#     #                                                       --data_path yahma/alpaca-cleaned --output_dir tune_log/${tune_ckpt_path} \
#     #                                                       --wandb_project DistillPrune --lora_r 8 --num_epochs 2 \
#     #                                                       --learning_rate 1e-4 --batch_size 64
#     # echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Prune and Post-Training."

#     # # Evaluating
#     # echo "[${name} - Sparsity: ${sparsity}] [START] - Start Evaluation on GPU ${gpu_id}"
#     # echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate.sh ${base_model} "" prune_log/${name}_s${sparsity}_block 0
#     # echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate.sh ${base_model} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
#     # echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Evaluation"
#     # echo "[${name} - Sparsity: ${sparsity}] [INFO] - The pruned model is at prune_log/${prune_ckpt_path}/pytorch_model.bin, and the recovery weight is at tune_log/${tune_ckpt_path}/"
# }

# # Number of available GPUs; adjust if needed
# num_gpus=4

# # Main loop to run each model
# for model in "${models[@]}"; do
#     echo "Starting processing for model: ${model}"
    
#     for i in "${!sparsity_values[@]}"; do
#         sparsity="${sparsity_values[$i]}"
#         gpu_id=$((i % num_gpus))  # Cycle through available GPUs
#         # Run each sparsity level concurrently for the current model
#         (
#             run_pipeline "$model" "$sparsity" "$gpu_id"
#         ) &
#     done
    
#     # Wait for all sparsity jobs for the current model to finish
#     wait
#     echo "Completed processing for model: ${model}"
# done

# echo "All processing completed."

# #!/bin/bash

# # Exit immediately if a command exits with a non-zero status
# set -e

# # Function to handle errors and output error message
# trap 'echo "[ERROR] A command failed on line $LINENO. Exiting."' ERR

# # Define models and their properties
# models=(
#     "meta-llama/Llama-3.1-8B"
#     # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# )

# sparsity_values=("0.10" "0.25" "0.50" "0.75")

# # Function to prune, fine-tune, and evaluate a model for a single sparsity level
# run_pipeline() {
#     local base_model=$1
#     local sparsity=$2
#     local gpu_id=$3
#     local name=${base_model##*/}  # Extract the model's name from path

#     prune_ckpt_path="${name}_s${sparsity}_block"
#     tune_ckpt_path="${name}_s${sparsity}_block"

#     # Pruning
#     echo "[${name} - Sparsity: ${sparsity}] [START] - Start Pruning Model on GPU ${gpu_id}"
#     echo y | CUDA_VISIBLE_DEVICES=${gpu_id} python llama3.py --base_model ${base_model} \
#                                                     --pruning_ratio ${sparsity} \
#                                                     --device cpu --eval_device cuda \
#                                                     --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
#                                                     --block_attention_layer_start 4 --block_attention_layer_end 28 \
#                                                     --save_ckpt_log_name ${prune_ckpt_path} \
#                                                     --pruner_type taylor --taylor param_first \
#                                                     --max_seq_len 2048 \
#                                                     --save_model #--test_before_train --test_after_train 
#     echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Pruning Model"

#     # # Fine-tuning
#     # echo "[${name} - Sparsity: ${sparsity}] [START] - Start Tuning on GPU ${gpu_id}"
#     # echo y | CUDA_VISIBLE_DEVICES=${gpu_id} python post_training.py --prune_model prune_log/${prune_ckpt_path}/pytorch_model.bin \
#     #                                                       --data_path yahma/alpaca-cleaned --output_dir tune_log/${tune_ckpt_path} \
#     #                                                       --wandb_project DistillPrune --lora_r 8 --num_epochs 2 \
#     #                                                       --learning_rate 1e-4 --batch_size 64
#     # echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Prune and Post-Training."

#     # # Evaluating
#     # echo "[${name} - Sparsity: ${sparsity}] [START] - Start Evaluation on GPU ${gpu_id}"
#     # echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate.sh ${base_model} "" prune_log/${name}_s${sparsity}_block 0
#     # echo y | CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/evaluate.sh ${base_model} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
#     # echo "[${name} - Sparsity: ${sparsity}] [FINISH] - Finish Evaluation"
#     # echo "[${name} - Sparsity: ${sparsity}] [INFO] - The pruned model is at prune_log/${prune_ckpt_path}/pytorch_model.bin, and the recovery weight is at tune_log/${tune_ckpt_path}/"
# }

# # Number of available GPUs; adjust if needed
# num_gpus=4

# # Main loop to run each model
# for model in "${models[@]}"; do
#     echo "Starting processing for model: ${model}"
    
#     for i in "${!sparsity_values[@]}"; do
#         sparsity="${sparsity_values[$i]}"
#         gpu_id=$((i % num_gpus))  # Cycle through available GPUs
#         # Run each sparsity level concurrently for the current model
#         (
#             run_pipeline "$model" "$sparsity" "$gpu_id"
#         ) &
#     done
    
#     # Wait for all sparsity jobs for the current model to finish
#     wait
#     echo "Completed processing for model: ${model}"
# done

# echo "All processing completed."