export CUDA_VISIBLE_DEVICES="0,1,2,3"




export model='meta-llama/Llama-3.2-3B-Instruct'
#'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
export outdir='Llama-3.2-3B-Instruct'
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
