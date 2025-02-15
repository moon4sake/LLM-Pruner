export CUDA_VISIBLE_DEVICES="0,1,2,3"

export model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
export outdir='deepseek-8b'


python s1_score.py --model ${model} \
    --tokenizer ${model} \
    --architecture_name llama \
    --data_file ./data/gsm8k_test.csv \
    --outdir ${outdir} \
    --tensor_parallel_size 4 \
