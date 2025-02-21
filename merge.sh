#!/bin/bash

# List of model names to process
MODEL_NAMES=(
    # "DeepSeek-R1-Distill-Qwen-1.5B_s0.10_channel"
    # "DeepSeek-R1-Distill-Qwen-1.5B_s0.25_channel"
    "DeepSeek-R1-Distill-Qwen-1.5B_s0.50_block"
    "DeepSeek-R1-Distill-Qwen-1.5B_s0.75_block"
    # "DeepSeek-R1-Distill-Llama-8B_s0.10_block"
    # "DeepSeek-R1-Distill-Llama-8B_s0.10_channel"
    # "DeepSeek-R1-Distill-Llama-8B_s0.50_channel"
    # "DeepSeek-R1-Distill-Llama-8B_s0.75_channel"
)

# Base directories for model and adapter paths
BASE_PRUNE_LOG_DIR="./prune_log/"
BASE_TUNE_LOG_DIR="./tune_log/"
BASE_HF_REPO="moon4sake"

# Hugging Face token
MY_HF_TOKEN="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"

# Loop through each specified model name
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Processing model: $MODEL_NAME"

    # Construct paths based on model name
    BASE_MODEL_PATH="${BASE_PRUNE_LOG_DIR}${MODEL_NAME}/pytorch_model.bin"
    ADAPTER_PATH="${BASE_TUNE_LOG_DIR}${MODEL_NAME}"
    FINAL_DIR="${BASE_TUNE_LOG_DIR}${MODEL_NAME}"
    REPO_DIR="${BASE_HF_REPO}/${MODEL_NAME}"

    # Run the Python script for each model
    python3 merge_upload.py \
        --base_model_name_or_path "$BASE_MODEL_PATH" \
        --peft_model_path "$ADAPTER_PATH" \
        --output_dir "$FINAL_DIR" \
        --repo_path "$REPO_DIR" \
        --hf_token "$MY_HF_TOKEN"
done

echo "All specified models processed successfully."


# # model
# NAME=Llama-3.1-8B_s0.10_channel
# base_model_path=prune_log/${NAME}/pytorch_model.bin
# adapter_path=tune_log/${NAME}
# # merged output
# final_dir=tune_log/${NAME}
# # repository directory
# repo_dir=moon4sake/${NAME}
# my_hf_tokens=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

# python3 merge_upload.py \
#     --base_model_name_or_path $base_model_path \
#     --peft_model_path $adapter_path \
#     --output_dir $final_dir \
#     --repo_path $repo_dir \
#     --hf_token $my_hf_tokens


# python3 merge.py \
#     --base_model_name_or_path $base_model_path \
#     --peft_model_path $adapter_path \
#     --output_dir $final_dir \
#     --hf_token $my_hf_tokens

# python3 upload_model.py \
#     --base_model $final_dir \
#     --repo_path $repo_dir \
#     --hf_token $my_hf_tokens

