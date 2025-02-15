# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"

# Arguments
pruning_ratio=0.13  # 0.13, 0.40, 0.67, 0.93

# Run the script
python ${SCRIPT_DIR}/prune_width.py --pruning_ratio ${pruning_ratio} \
                 --device cuda --eval_device cuda \
                 --base_model Qwen/Qwen2.5-1.5B-Instruct \
                 --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 28 \
                 --block_attention_layer_start -1 --block_attention_layer_end -1 \
                 --save_ckpt_log_name Qwen2.5-1.5B_mlp_${pruning_ratio} \
                 --pruner_type l2 \
                 --max_seq_len 2048 \
                 --save_model  # --test_after_train --test_before_train
