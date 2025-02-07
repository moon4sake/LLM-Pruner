## Model-list: 
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
## Qwen/Qwen2.5-Math-1.5B 
## meta-llama/Llama-3.1-8B 
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B

base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
name="DeepSeek-R1-Distill-Llama-8B_s0.10_block"

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
                            ${base_model} "" prune_log/${name} 0

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
                            ${base_model} tune_log/${name} prune_log/${name} 1400
