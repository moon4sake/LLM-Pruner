#!/bin/bash

## Model-list: 
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
## Qwen/Qwen2.5-Math-1.5B 
## meta-llama/Llama-3.1-8B 
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B


# Define base model and initial name
base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
name="DeepSeek-R1-Distill-Llama-8B"

# Define an array of sparsity values you want to use
sparsity_values=("0.10" "0.25" "0.50")

# Iterate over sparsity values and run on different GPUs
for i in {0..2}; do
  sparsity="${sparsity_values[$i]}"
  
  # Use a subshell to run each evaluation in the background
  (
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_block 0
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
  ) &
done

# Wait for all background jobs to finish
wait

echo "[${name}] All evaluations completed."


# Define base model and initial name
base_model="meta-llama/Llama-3.1-8B"
name="Llama-3.1-8B"

# Define an array of sparsity values you want to use
sparsity_values=("0.10" "0.25" "0.50")

# Iterate over sparsity values and run on different GPUs
for i in {0..2}; do
  sparsity="${sparsity_values[$i]}"
  
  # Use a subshell to run each evaluation in the background
  (
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_block 0
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
  ) &
done

# Wait for all background jobs to finish
wait

echo "[${name}] All evaluations completed."

# Define base model and initial name
base_model="Qwen/Qwen2.5-Math-1.5B"
name="Qwen2.5-Math-1.5B"

# Define an array of sparsity values you want to use
sparsity_values=("0.10" "0.25" "0.50")

# Iterate over sparsity values and run on different GPUs
for i in {0..2}; do
  sparsity="${sparsity_values[$i]}"
  
  # Use a subshell to run each evaluation in the background
  (
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_block 0
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
  ) &
done

# Wait for all background jobs to finish
wait

echo "[${name}] All evaluations completed."


# Define base model and initial name
base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
name="DeepSeek-R1-Distill-Qwen-1.5B"

# Define an array of sparsity values you want to use
sparsity_values=("0.10" "0.25" "0.50")

# Iterate over sparsity values and run on different GPUs
for i in {0..2}; do
  sparsity="${sparsity_values[$i]}"
  
  # Use a subshell to run each evaluation in the background
  (
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_block 0
    echo y | CUDA_VISIBLE_DEVICES=$i bash scripts/evaluate.sh ${base_model} ${sparsity} tune_log/${name}_s${sparsity}_block prune_log/${name}_s${sparsity}_block 1400
  ) &
done

# Wait for all background jobs to finish
wait

echo "[${name}] All evaluations completed."