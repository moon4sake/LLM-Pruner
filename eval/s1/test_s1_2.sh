export CUDA_VISIBLE_DEVICES="4,5,6,7"

# export model='moon4sake/Llama-3.1-8B_s0.10_channel'
# export outdir='moon4sake_Llama-3.1-8B_s0.10_channel'


# python s1_score.py --model ${model} \
#     --tokenizer 'meta-llama/Llama-3.1-8B-Instruct' \
#     --architecture_name llama \
#     --data_file ./data/gsm8k_test.csv \
#     --outdir ${outdir} \
#     --tensor_parallel_size 4 \



# export model='moon4sake/Llama-3.1-8B_s0.25_channel'
# export outdir='moon4sake_Llama-3.1-8B_s0.25_channel'


# python s1_score.py --model ${model} \
#     --tokenizer 'meta-llama/Llama-3.1-8B-Instruct' \
#     --architecture_name llama \
#     --data_file ./data/gsm8k_test.csv \
#     --outdir ${outdir} \
#     --tensor_parallel_size 4 \




# export model='moon4sake/Llama-3.1-8B_s0.50_channel'
# export outdir='moon4sake_Llama-3.1-8B_s0.50_channel'


# python s1_score.py --model ${model} \
#     --tokenizer 'meta-llama/Llama-3.1-8B-Instruct' \
#     --architecture_name llama \
#     --data_file ./data/gsm8k_test.csv \
#     --outdir ${outdir} \
#     --tensor_parallel_size 4 \





# export model='moon4sake/Llama-3.1-8B_s0.75_channel'
# export outdir='moon4sake_Llama-3.1-8B_s0.75_channel'


# python s1_score.py --model ${model} \
#     --tokenizer 'meta-llama/Llama-3.1-8B-Instruct' \
#     --architecture_name llama \
#     --data_file ./data/gsm8k_test.csv \
#     --outdir ${outdir} \
#     --tensor_parallel_size 8 \




# export model='meta-llama/Llama-3.1-8B '
# export outdir='llma8b'


# python s1_score.py --model ${model} \
#     --tokenizer 'meta-llama/Llama-3.1-8B-Instruct' \
#     --architecture_name llama \
#     --data_file ./data/gsm8k_test.csv \
#     --outdir ${outdir} \
#     --tensor_parallel_size 8 \




export model='moon4sake/Qwen2.5-Math-1.5B_s0.10_channel'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.10_channel'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \




export model='moon4sake/Qwen2.5-Math-1.5B_s0.25_channel'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.25_channel'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \



export model='moon4sake/Qwen2.5-Math-1.5B_s0.50_channel'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.50_channel'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \


export model='moon4sake/Qwen2.5-Math-1.5B_s0.75_channel'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.75_channel'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \


export model='Qwen/Qwen2.5-Math-1.5B'
export outdir='Qwen2.5-Math-1.5B '


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \



export model='Qwen/Qwen2.5-Math-1.5B  '
export outdir='Qwen2.5-Math-1.5B '

python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \




## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
## Qwen/Qwen2.5-Math-1.5B
## meta-llama/Llama-3.1-8B
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B