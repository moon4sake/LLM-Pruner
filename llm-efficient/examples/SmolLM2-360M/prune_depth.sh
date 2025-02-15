# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"

# Arguments
pruning_ratio=0.1  # 0.1, 0.3, 0.5, 0.7

# Run the script
python ${SCRIPT_DIR}/depth_pruning/prune_depth.py \
                 --original_model_path HuggingFaceTB/SmolLM2-360M-Instruct \
                 --config_file_path HuggingFaceTB/SmolLM2-360M-Instruct \
                 --skip_file_path output/SmolLM2-360M_depth_${pruning_ratio}/remove_${pruning_ratio}.json \
                 --model_save_path output/SmolLM2-360M_depth_${pruning_ratio}
