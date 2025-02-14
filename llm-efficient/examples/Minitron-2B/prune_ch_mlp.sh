# Private variables
SCRIPT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"scripts"
OUTPUT_DIR=$(dirname $(dirname $(dirname $(realpath "$0"))))/"output"

BASE_MODEL=Mistral-Nemo-Minitron-2B-128k-Instruct

# Arguments
pruning_ratio=0.13  # 0.13, 0.26, 0.40, 0.60

# Run the script
python ${SCRIPT_DIR}/prune_width.py --pruning_ratio ${pruning_ratio} \
                 --device cuda --eval_device cuda \
                 --base_model ${BASE_MODEL} \
                 --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 34 \
                 --block_attention_layer_start -1 --block_attention_layer_end -1 \
                 --save_ckpt_log_name Minitron-2B_mlp_${pruning_ratio} \
                 --pruner_type taylor \
                 --max_seq_len 2048 \
                 --save_model  # --test_after_train --test_before_train

cp ${BASE_MODEL}/special_tokens_map.json ${OUTPUT_DIR}/Minitron-2B_mlp_${pruning_ratio}/special_tokens_map.json
cp ${BASE_MODEL}/tokenizer_config.json ${OUTPUT_DIR}/Minitron-2B_mlp_${pruning_ratio}/tokenizer_config.json
cp ${BASE_MODEL}/tokenizer.json ${OUTPUT_DIR}/Minitron-2B_mlp_${pruning_ratio}/tokenizer.json

python ${SCRIPT_DIR}/prune_width.py --pruning_ratio ${pruning_ratio} \
                 --device cuda --eval_device cuda \
                 --base_model ${OUTPUT_DIR}/Minitron-2B_mlp_${pruning_ratio} \
                 --channel_wise \
                 --block_wise --block_mlp_layer_start -1 --block_mlp_layer_end -1 \
                 --block_attention_layer_start -1 --block_attention_layer_end -1 \
                 --save_ckpt_log_name Minitron-2B_ch_mlp_${pruning_ratio} \
                 --pruner_type taylor \
                 --max_seq_len 2048 \
                 --save_model  # --test_after_train --test_before_train
