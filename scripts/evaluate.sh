#!/bin/bash
export PYTHONPATH='.'

# base_model=$1 # e.g., decapoda-research/llama-7b-hf
# tune_ckpt_name=$2 
# prune_ckpt=$3
# epochs=("${@:4}")

# for epoch in "${epochs[@]}"; 
# do
#     cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
#     cp $tune_ckpt_name/adapter_model.bin $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin
#     mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

#     tune_id="${tune_ckpt_name##*/}"
#     python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}_$epoch.json --no_cache
# done

base_model=$1
sparsity=$2
tune_ckpt_name=$3
prune_ckpt=$4
epochs=("${@:5}")

# Check condition for only pruned model evaluation
if [[ -z "$tune_ckpt_name" || ( "${#epochs[@]}" -eq 1 && "${epochs[0]}" -eq 0 ) ]]; then
    # Run evaluation with only the pruned model
    echo "Evaluating pruned model without fine-tuning..."
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${base_model}_s${sparsity}_pruned_model.json --no_cache
else
    # Run evaluation with both the pruned and fine-tuned model
    echo "Evaluating pruned and fine-tuned model..."
    for epoch in "${epochs[@]}"; do
        cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
        cp $tune_ckpt_name/adapter_model.bin $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin
        mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

        tune_id="${tune_ckpt_name##*/}"
        python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}_s${sparsity}_$epoch.json --no_cache
    done
fi