# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"

# Arguments
pruning_ratio=0.1  # 0.1, 0.3, 0.5, 0.7

# Run the script
python ${SCRIPT_DIR}/depth_pruning/compute_importance_score.py \
                 --base_model Mistral-Nemo-Minitron-2B-128k-Instruct \
                 --output_folder output/Minitron-2B_depth_${pruning_ratio} \
                 --batch_size 32 \
                 --percent ${pruning_ratio}
