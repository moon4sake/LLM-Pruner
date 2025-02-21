CUDA_VISIBLE_DEVICES=0 python llama3.py --pruning_ratio 0.25 \
                                        --device cuda --eval_device cuda \
                                        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
                                        --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
                                        --block_attention_layer_start 4 --block_attention_layer_end 30 \
                                        --save_ckpt_log_name llama3_prune \
                                        --pruner_type taylor --taylor param_first \
                                        --max_seq_len 2048 \
                                        --test_after_train --test_before_train --save_model 


source activate && conda activate pruning2 && clear