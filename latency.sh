#!/bin/bash

# List of Hugging Face models to evaluate
MODELS=(
    "meta-llama/Llama-3.1-8B "
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    ""
)

# Run the evaluation for each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    python test_speedup.py --base_model "$MODEL" \
                           --model_type pretrain \
                           --output_dir ./visualize/latency/
done