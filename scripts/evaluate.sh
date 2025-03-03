#!/bin/bash
export PYTHONPATH='.'

eval_type=$1
base_model=$2
sparsity=$3
tune_ckpt_name=$4
prune_ckpt=$5
epochs=$6
num_fewshot=$7
suffix=$7

# Check condition for only pruned model evaluation
if [[ -z "$tune_ckpt_name" || ( "${#epochs[@]}" -eq 1 && "${epochs[0]}" -eq 0 ) ]]; then    
    # Run evaluation with the pre-trained model
    echo "Evaluating the pre-trained model..."
    if [ "$eval_type" == "simple" ]; then
        python lm-evaluation-harness/main.py --model hf --model_args pretrained=$base_model,trust_remote_code=True --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${base_model}_pre-trained_model_${num_fewshot}shot.json --num_fewshot ${num_fewshot} --no_cache
    elif [ "$eval_type" == "math" ]; then
        mkdir -p results/gsm8k
        python lm-evaluation-harness/main.py --model hf --model_args pretrained=$base_model,trust_remote_code=True --tasks gsm8k_cot_llama --device cuda:0 --output_path results/gsm8k/${base_model}_pre-trained_model.json --fewshot_as_multiturn --apply_chat_template
    fi
else
    # Run evaluation with both the pruned and fine-tuned model
    echo "Evaluating pruned and fine-tuned model..."
    for epoch in "${epochs[@]}"; do
        cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
        cp $tune_ckpt_name/adapter_model.bin $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin
        mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

        tune_id="${tune_ckpt_name##*/}"
        if [ "$eval_type" == "simple" ]; then
            python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}_s${sparsity}_${epoch}_${num_fewshot}shot_${suffix}.json --num_fewshot ${num_fewshot} --no_cache
        elif [ "$eval_type" == "math" ]; then
            mkdir -p results/gsm8k
            python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks gsm8k_cot_llama --device cuda:0 --output_path results/gsm8k/${tune_id}_s${sparsity}${suffix}.json --fewshot_as_multiturn --apply_chat_template
        fi
    done
fi          