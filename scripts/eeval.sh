#!/bin/bash

## Model-list: 
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
## Qwen/Qwen2.5-Math-1.5B 
## meta-llama/Llama-3.1-8B 
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B


# Define base model and initial name
base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
name="DeepSeek-R1-Distill-Qwen-7B"

# Define an array of sparsity values you want to use
sparsity_values=("0.10" "0.25" "0.50" "0.75")

# Iterate over sparsity values and run on different GPUs
for i in {0..3}; do
  sparsity="${sparsity_values[$i]}"
  
  # Use a subshell to run each evaluation in the background
  (
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} "" prune_log/${name}_s${sparsity} 0
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} tune_log/${name}_s${sparsity} prune_log/${name}_s${sparsity} 1400
  ) &
done

# Wait for all background jobs to finish
wait

echo "All evaluations completed."
