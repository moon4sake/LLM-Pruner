## Model-list: 
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
## Qwen/Qwen2.5-Math-1.5B 
## meta-llama/Llama-3.1-8B 
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# name="DeepSeek-R1-Distill-Qwen-1.5B"
# sparsity_list="0.10 0.25 0.50"

# for sparsity in ${sparsity_list}
# do
#     CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
#                                 ${base_model} "" prune_log/${name} 0

#     CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
#                                 ${base_model} tune_log/${name} prune_log/${name} 1400
# done

### TODO: Dense model evaluation ###
# base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# name="DeepSeek-R1-Distill-Llama-8B"
# CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
#                                 ${base_model} "0.00" "" prune_log/${name}_s${sparsity}_channel 0
####################################

base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
name="DeepSeek-R1-Distill-Qwen-1.5B"
sparsity_list="0.10 0.25 0.50"

for sparsity in ${sparsity_list}
do
    CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
                                ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_channel 0

    CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
                                ${base_model} ${sparsity} tune_log/${name}_s${sparsity}_channel prune_log/${name}_s${sparsity}_channel 1400
done


base_model="Qwen/Qwen2.5-Math-1.5B"
name="Qwen2.5-Math-1.5B"
sparsity_list="0.10 0.25 0.50"

for sparsity in ${sparsity_list}
do
    CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
                                ${base_model} ${sparsity} "" prune_log/${name}_s${sparsity}_channel 0

    CUDA_VISIBLE_DEVICES=3 bash scripts/evaluate.sh \
                                ${base_model} ${sparsity} tune_log/${name}_s${sparsity}_channel prune_log/${name}_s${sparsity}_channel 1400
done