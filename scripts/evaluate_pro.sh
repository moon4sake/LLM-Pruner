#!/bin/bash
export PYTHONPATH='.'

base_model=$1
sparsity=$2
tune_ckpt_name=$3
prune_ckpt=$4
epoch=$5
num_fewshot=$6
suffix=$7

# Check condition for only pruned model evaluation
if [[ -z "$tune_ckpt_name" || ( "${#epochs[@]}" -eq 1 && "${epochs[0]}" -eq 0 ) ]]; then    
    # Run evaluation with the pre-trained model
    echo "Evaluating the pre-trained model..."
    # MATH
    mkdir -p results/gsm8k
    mkdir -p results/math_hard
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pretrained=$base_model,trust_remote_code=True --tasks gsm8k_cot_llama --device cuda:0 --output_path results/gsm8k/${base_model}_pre-trained_model.json --fewshot_as_multiturn --apply_chat_template
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pretrained=$base_model,trust_remote_code=True --tasks leaderboard_math_hard --device cuda:0 --output_path results/math_hard/${base_model}_pre-trained_model.json
    # Code
    mkdir -p results/humaneval
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pretrained=$base_model,trust_remote_code=True --tasks humaneval --device cuda:0 --output_path results/humaneval/${base_model}_pre-trained_model.json
    # Instruction following
    mkdir -p results/ifeval
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pretrained=$base_model,trust_remote_code=True --tasks leaderboard_ifeval --device cuda:0 --output_path results/ifeval/${base_model}_pre-trained_model.json
    # Science
    mkdir -p results/gpqa
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pretrained=$base_model,trust_remote_code=True --tasks leaderboard_gpqa --device cuda:0 --output_path results/gpqa/${base_model}_pre-trained_model.json
else
    # Run evaluation with both the pruned and fine-tuned model
    echo "Evaluating pruned and fine-tuned model..."
    cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
    cp $tune_ckpt_name/adapter_model.bin $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin
    mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

    name=${base_model##*/}
    prune_ckpt_path="prune_log/${name}_s${sparsity}_block_all_global/pytorch_model.bin"
    tune_ckpt_path="tune_log/${name}_s${sparsity}_block_all_global/checkpoint-${epoch}"

    tune_id="${tune_ckpt_name##*/}"
    # Math
    mkdir -p results/gsm8k
    mkdir -p results/math_hard
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pruned=${prune_ckpt_path},adapter=${tune_ckpt_path} --tasks gsm8k_cot_llama --device cuda:0 --output_path results/gsm8k/${tune_id}_s${sparsity}${suffix}.json --fewshot_as_multiturn --apply_chat_template
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pruned=${prune_ckpt_path},adapter=${tune_ckpt_path} --tasks leaderboard_math_hard --device cuda:0 --output_path results/math_hard/${tune_id}_s${sparsity}${suffix}.json
    # Code
    mkdir -p results/humaneval
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pruned=${prune_ckpt_path},adapter=${tune_ckpt_path} --tasks humaneval --device cuda:0 --output_path results/humaneval/${tune_id}_s${sparsity}${suffix}.json
    # Instruction following
    mkdir -p results/ifeval
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pruned=${prune_ckpt_path},adapter=${tune_ckpt_path} --tasks leaderboard_ifeval --device cuda:0 --output_path results/ifeval/${tune_id}_s${sparsity}${suffix}.json
    # Science
    mkdir -p results/gpqa
    python lm-evaluation-harness_v1/lm_eval/__main__.py --model local --model_args pruned=${prune_ckpt_path},adapter=${tune_ckpt_path} --tasks leaderboard_gpqa --device cuda:0 --output_path results/gpqa/${tune_id}_s${sparsity}${suffix}.json
fi          