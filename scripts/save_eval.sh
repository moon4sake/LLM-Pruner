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
  "moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.25_block"
  # "meta-llama/Meta-Llama-3-8B-Instruct"
    # Uncomment and add more models if needed
    # "moon4sake/Qwen2.5-Math-1.5B_s0.25_channel"
    # "moon4sake/Qwen2.5-Math-1.5B_s0.50_channel"
    # "moon4sake/Qwen2.5-Math-1.5B_s0.75_channel"
    # "Qwen/Qwen2.5-Math-1.5B"
    # "moon4sake/Qwen2.5-Math-1.5B_s0.10_channel"
)

# Define tasks and few-shot parameter
TASKS="openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,gsm8k"
NUM_FEWSHOT=5

# Number of GPUs available and their IDs
GPUS=(0) # 1 2 3)

# Loop over available GPUs
for i in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$i]}"
    MODEL="${MODELS[$i]}"  # Assuming one model per GPU for simplicity
    OUTPUT_FILE="eval_results_new/${MODEL}_results.json"

    echo "Starting evaluation for model: $MODEL on GPU: $GPU_ID"
    
    # Run the evaluation command on the selected GPU
    CUDA_VISIBLE_DEVICES="$GPU_ID" lm-eval --model_args pretrained="$MODEL",ignore_mismatched_sizes=True \
                                           --tasks "$TASKS" \
                                           --device cuda \
                                           --num_fewshot "$NUM_FEWSHOT" \
                                           --output_path "$OUTPUT_FILE" &

done

# Wait for all processes to complete
wait

echo "Evaluation completed on all GPUs."