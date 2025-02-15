# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"

# Arguments
pruning_ratio=0.5
mlp_pruning_ratio=0.00  # 0.00, 0.13, 0.67

# Run the script
python ${SCRIPT_DIR}/prune_width.py --pruning_ratio ${pruning_ratio} --mlp_pruning_ratio ${mlp_pruning_ratio} \
                 --device cuda --eval_device cuda \
                 --base_model Qwen/Qwen2.5-1.5B-Instruct \
                 --channel_wise \
                 --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 28 \
                 --block_attention_layer_start 0 --block_attention_layer_end 28 \
                 --save_ckpt_log_name Qwen2.5-1.5B_ch_mlp_attn_${pruning_ratio}_${mlp_pruning_ratio} \
                 --pruner_type l2 \
                 --max_seq_len 2048 \
                 --save_model  # --test_after_train --test_before_train
