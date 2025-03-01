## Model-list: 
## deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
## Qwen/Qwen2.5-Math-1.5B 
## meta-llama/Llama-3.1-8B 
## deepseek-ai/DeepSeek-R1-Distill-Llama-8B

export CUDA_VISIBLE_DEVICES="1,2,3,4"

export model='Qwen/Qwen2.5-Math-1.5B'
export outdir='results/s1/Qwen2.5-Math-1.5B'

python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name qwen \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \
