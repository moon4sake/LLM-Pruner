#!/bin/bash

# List of Hugging Face models to evaluate
MODELS=(
    "Qwen/Qwen2.5-Math-1.5B"
    "moon4sake/Qwen2.5-Math-1.5B_s0.25_channel"
    "moon4sake/Qwen2.5-Math-1.5B_s0.50_channel"
    "moon4sake/Qwen2.5-Math-1.5B_s0.10_channel"
    "moon4sake/Qwen2.5-Math-1.5B_s0.75_channel"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.10_channel"
    "moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.25_channel"
    "moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.50_channel"
    "moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.75_channel"
)

# Run the evaluation for each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    python test_speedup.py --base_model "$MODEL" \
                           --model_type pretrain \
                           --output_dir ./visualize/latency/
done