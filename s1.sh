export CUDA_VISIBLE_DEVICES="4,5,6,7"




export model='moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.10_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Llama-8B_s0.10_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4




export model='moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.25_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Llama-8B_s0.25_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4




export model='moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.50_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Llama-8B_s0.50_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'평가

python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4




export model='moon4sake/DeepSeek-R1-Distill-Llama-8B_s0.75_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Llama-8B_s0.75_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4



export model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='deepseek_ai_DeepSeek-R1-Distill-Llama-8B'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \





export model='moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.10_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Qwen-1.5B_s0.10_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4




export model='moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.25_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Qwen-1.5B_s0.25_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4




export model='moon4sake/DeepSeek-R1-Distill-Qwen-1.5B_s0.50_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Qwen-1.5B_s0.50_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4




export model='moon4sake_DeepSeek-R1-Distill-Qwen-1.5B_s0.75_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_DeepSeek-R1-Distill-Qwen-1.5B_s0.75_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4



export model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='deepseek_ai_DeepSeek-R1-Distill-Qwen-1.5B'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4



export model='moon4sake/Qwen2.5-Math-1.5B_s0.10_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.10_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4


export model='moon4sake/Qwen2.5-Math-1.5B_s0.25_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.25_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4


export model='moon4sake/Qwen2.5-Math-1.5B_s0.50_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.50_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4


export model='moon4sake/Qwen2.5-Math-1.5B_s0.75_channel'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='moon4sake_Qwen2.5-Math-1.5B_s0.75_channel'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4


export model='Qwen/Qwen2.5-Math-1.5B'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='Qwen_Qwen2.5-Math-1.5B'
# 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name r1 \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4







## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
## Qwen/Qwen2.5-Math-1.5B
## meta-llama/Llama-3.1-8B
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B
