# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"

# Arguments
pruning_ratio=0.15  # 0.15, 0.42, 0.69, 0.98

# Run the script
python ${SCRIPT_DIR}/prune_width.py --pruning_ratio ${pruning_ratio} \
                 --device cuda --eval_device cuda \
                 --base_model Mistral-Nemo-Minitron-2B-128k-Instruct \
                 --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 34 \
                 --block_attention_layer_start -1 --block_attention_layer_end -1 \
                 --save_ckpt_log_name Minitron-2B_mlp_${pruning_ratio} \
                 --pruner_type taylor \
                 --max_seq_len 2048 \
                 --save_model  # --test_after_train --test_before_train
