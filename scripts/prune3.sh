prune_ckpt_path='DeepSeek-R1-Distill-Llama-8B_s0.55'
tune_ckpt_path='DeepSeek-R1-Distill-Llama-8B_s0.5'

echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python llama3.py --base_model meta-llama/Llama-3.1-8B \
#                                         --pruning_ratio 0.25 \
#                                         --device cuda --eval_device cuda \
#                                         --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
#                                         --block_attention_layer_start 4 --block_attention_layer_end 30 \
#                                         --save_ckpt_log_name $prune_ckpt_path \
#                                         --pruner_type taylor --test_after_train --taylor param_first \
#                                         --save_model
CUDA_VISIBLE_DEVICES=3 python llama3.py --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                                        --pruning_ratio 0.5 \
                                        --device cuda --eval_device cuda \
                                        --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
                                        --block_attention_layer_start 4 --block_attention_layer_end 30 \
                                        --save_ckpt_log_name $prune_ckpt_path \
                                        --pruner_type taylor --taylor param_first \
                                        --max_seq_len 2048 \
                                        --test_after_train --save_model #--test_before_train
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=3 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"



