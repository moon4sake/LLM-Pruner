prune_ckpt_path='Qwen2.5-Math-7B_s0.10'
tune_ckpt_path='Qwen2.5-Math-7B_s0.10'

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=2 python llama3.py --base_model Qwen/Qwen2.5-Math-7B \
                                        --pruning_ratio 0.10 \
                                        --device cuda --eval_device cuda \
                                        --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
                                        --block_attention_layer_start 4 --block_attention_layer_end 30 \
                                        --save_ckpt_log_name $prune_ckpt_path \
                                        --pruner_type taylor --taylor param_first \
                                        --max_seq_len 2048 \
                                        --test_after_train --save_model #--test_before_train
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=2 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"



