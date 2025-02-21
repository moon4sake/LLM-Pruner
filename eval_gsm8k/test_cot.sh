export CUDA_VISIBLE_DEVICES="0,1,2,3"



export model='/mnt/home/sy/MaskedThought/cot_120k_answer_nonzero/checkpoint-6030'
export outdir='cot_120k_nonzero'

python eval_gsm8k.py --model ${model} --data_file ./data/test/GSM8K_test.jsonl --outdir ${outdir} --tensor_parallel_size 4

export model='/mnt/home/sy/MaskedThought/cot_120k_question_conf75/checkpoint-4542'
export outdir='cot_120k_question_conf75'

python eval_gsm8k.py --model ${model} --data_file ./data/test/GSM8K_test.jsonl --outdir ${outdir} --tensor_parallel_size 4

export model='/mnt/home/sy/MaskedThought/cot_120k_answer_above4/checkpoint-3228'
export outdir='cot_120k_ans_confabove4'

python eval_gsm8k.py --model ${model} --data_file ./data/test/GSM8K_test.jsonl --outdir ${outdir} --tensor_parallel_size 4





