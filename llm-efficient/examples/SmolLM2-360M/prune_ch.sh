# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"

# Arguments
pruning_ratio=0.203  # 0.203, 0.380, 0.557, 0.734

# Run the script
python ${SCRIPT_DIR}/prune_width.py --pruning_ratio ${pruning_ratio} \
                 --device cuda --eval_device cuda \
                 --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
                 --channel_wise \
                 --save_ckpt_log_name SmolLM2-360M_ch_${pruning_ratio} \
                 --pruner_type l2 \
                 --max_seq_len 2048 \
                 --save_model  # --test_after_train --test_before_train
