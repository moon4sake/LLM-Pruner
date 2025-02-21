## pre-trained models
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
## Qwen/Qwen2.5-Math-1.5B 
## meta-llama/Llama-3.1-8B 
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B


#!/bin/bash

# Ensure the eval_results directory exists
mkdir -p eval_results

# List of Hugging Face models to evaluate
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
    # Uncomment and add more models if needed
    # "moon4sake/Qwen2.5-Math-1.5B_s0.25_channel"
    # "moon4sake/Qwen2.5-Math-1.5B_s0.50_channel"
    # "moon4sake/Qwen2.5-Math-1.5B_s0.75_channel"
    # "Qwen/Qwen2.5-Math-1.5B"
    # "moon4sake/Qwen2.5-Math-1.5B_s0.10_channel"
)

# Define tasks and few-shot parameter
TASKS=('gsm8k' 'gsm8k_cot' 'gsm8k_cot_zeroshot' 'gsm8k_cot_self_consistency')
NUM_FEWSHOT=0

# Number of GPUs available
NUM_GPUS=4
GPUS=(0 1 2 3)

# Loop over each model
for MODEL in "${MODELS[@]}"; do
  # Schedule tasks to run in parallel on different GPUs
  for ((j = 0; j < NUM_GPUS; j++)); do
    TASK=${TASKS[$j]}
    GPU_ID=${GPUS[$j]}
    
    echo "Starting evaluation for model: $MODEL on task: $TASK using GPU: $GPU_ID"
    OUTPUT_FILE="eval_results/${TASK}/$(basename ${MODEL})_results_${NUM_FEWSHOT}-shots.json"

    # Run the evaluation command for each task on a separate GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID lm-eval --model_args pretrained=${MODEL} \
                                         --tasks ${TASK} \
                                         --device cuda \
                                         --output_path "$OUTPUT_FILE" &
  done

  # Wait for all tasks for the current model to complete before moving to the next model
  wait
done

echo "All models evaluated on the specified tasks concurrently."





#!/bin/bash

# # Ensure the eval_results directory exists
# mkdir -p eval_results

# # List of Hugging Face models to evaluate
# MODELS=(
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#     # "moon4sake/Qwen2.5-Math-1.5B_s0.25_channel"
#     # "moon4sake/Qwen2.5-Math-1.5B_s0.50_channel"
#     # "moon4sake/Qwen2.5-Math-1.5B_s0.75_channel"
#     # "Qwen/Qwen2.5-Math-1.5B"
#     # "moon4sake/Qwen2.5-Math-1.5B_s0.10_channel"
# )

# # Define tasks and few-shot parameter
# # TASKS='gsm8k,gsm8k_cot,gsm8k_cot_zeroshot,gsm8k_cot_self_consistency' #"openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq"
# TASKS=('gsm8k' 'gsm8k_cot' 'gsm8k_cot_zeroshot' 'gsm8k_cot_self_consistency')
# NUM_FEWSHOT=0

# # Number of GPUs available
# NUM_GPUS=4
# GPUS=(0 1 2 3)

# # Loop over models in batches of four

# for TASK in "${TASKS[@]}"; do
#   for ((i = 0; i < ${#MODELS[@]}; i += NUM_GPUS)); do
#     # Schedule a batch of processes
#     for ((j = 0; j < NUM_GPUS; j++)); do
#       MODEL_INDEX=$((i + j))
      
#       # Check if we are out of models
#       if [ $MODEL_INDEX -ge ${#MODELS[@]} ]; then
#         break
#       fi

#       MODEL=${MODELS[$MODEL_INDEX]}
#       GPU_ID=${GPUS[$j]}
    
#       echo "Starting evaluation for model: $MODEL on GPU: $GPU_ID"
#       OUTPUT_FILE="eval_results/gsm8k/$(basename ${MODEL})_results_${TASK}.json"

#       # Run the evaluation command on the selected GPU
#       CUDA_VISIBLE_DEVICES=$GPU_ID lm-eval --model_args pretrained=${MODEL} \
#                                           --tasks ${TASK} \
#                                           --device cuda \
#                                           --output_path "$OUTPUT_FILE" &
#     done

#     # Wait for all processes in this batch to finish
#     wait
#   done
# done

# echo "All models evaluated on the specified tasks."


    # # Run the evaluation command on the selected GPU
    # CUDA_VISIBLE_DEVICES=$GPU_ID lm-eval --model_args pretrained=${MODEL} \
    #                                      --tasks ${TASKS} \
    #                                      --device cuda \
    #                                     #  --num_fewshot ${NUM_FEWSHOT} \
    #                                      --output_path "$OUTPUT_FILE" &