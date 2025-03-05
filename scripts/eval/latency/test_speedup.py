import os
import sys
import argparse
import time
import json

import torch
import numpy as np

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from ptflops import get_model_complexity_info

from LLMPruner.models.hf_llama.modeling_llama import LlamaAttention, LlamaRMSNorm
from LLMPruner.peft import PeftModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def LlamaAttention_counter_hook(module, input, output):
    flops = 0
    q_len = output[0].shape[1]
    linear_dim = output[0].shape[-1]
    num_heads = module.num_heads
    head_dim = module.head_dim

    rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
    attention_flops = num_heads * (q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim)
    linear_flops = 4 * (q_len * linear_dim * num_heads * head_dim)
    flops += rotary_flops + attention_flops + linear_flops
    module.__flops__ += int(flops)

def rmsnorm_flops_counter_hook(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    batch_flops *= 2
    module.__flops__ += int(batch_flops)

def main(args):
    if args.model_type == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    else:
        raise NotImplementedError

    def input_constructor(x):
        return {'input_ids': torch.ones(x).long().to(device)}

    model.to(device).eval()
    
    if device == "cuda":
        model.half()
        
    macs, params = get_model_complexity_info(
        model, 
        (1, 64,), 
        as_strings=True,
        input_constructor=input_constructor,
        print_per_layer_stat=True, 
        verbose=True,
        custom_modules_hooks={
            LlamaAttention: LlamaAttention_counter_hook,
            LlamaRMSNorm: rmsnorm_flops_counter_hook
        }
    )

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("GPU Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    # Load wikitext2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Measure latency
    input_length = 512  # Specify the input length
    start_time = time.time()

    with torch.no_grad():
        for sample in dataset:
            inputs = tokenizer(sample['text'], return_tensors="pt", max_length=input_length, truncation=True, padding="max_length")
            input_ids = inputs['input_ids'].to(device).long()  # Ensure input_ids are LongTensor
            attention_mask = inputs['attention_mask'].to(device)  # Move attention mask to the device

            # Assuming attention_mask is optionally required by the model
            model(input_ids=input_ids, attention_mask=attention_mask)
            
    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency for processing the test set: {latency:.2f} seconds")

    # Save results to JSON
    results = {
        "model_name": args.base_model,
        "macs": macs,
        "params": params,
        "latency": latency
    }

    output_path = os.path.join(args.output_dir, f"{os.path.basename(args.base_model)}_results.json")
    with open(output_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LLM Efficiency')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--model_type', type=str, required=True, help='choose from [pretrain, pruneLLM]')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True, help='directory to save results')

    args = parser.parse_args()
    main(args)


# import os
# import sys
# import argparse

# import torch
# import numpy as np

# import transformers
# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
# # from transformers.activations import SiLUActivation

# from ptflops import get_model_complexity_info
# from ptflops.pytorch_ops import bn_flops_counter_hook, pool_flops_counter_hook

# from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
# from LLMPruner.peft import PeftModel

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
# torch_version = int(torch.__version__.split('.')[1])

# def LlamaAttention_counter_hook(module, input, output):
#     # (1) Ignore past-key values
#     # (2) Assume there is no attention mask
#     # Input will be empty in some pytorch version. use output here since input.shape == output.shape
#     flops = 0
#     q_len = output[0].shape[1]
#     linear_dim = output[0].shape[-1]
#     num_heads = module.num_heads
#     head_dim = module.head_dim

#     rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
#     attention_flops = num_heads * (q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim) #QK^T + softmax + AttentionV
#     linear_flops = 4 * (q_len * linear_dim * num_heads * head_dim) # 4 for q, k, v, o. 
#     flops += rotary_flops + attention_flops + linear_flops
#     module.__flops__ += int(flops)

# def rmsnorm_flops_counter_hook(module, input, output):
#     input = input[0]

#     batch_flops = np.prod(input.shape)
#     batch_flops *= 2
#     module.__flops__ += int(batch_flops)

# def main(args):
#     if args.model_type == 'pretrain':
#         # tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
#         # model = LlamaForCausalLM.from_pretrained(
#         #     args.base_model,
#         #     low_cpu_mem_usage=True if torch_version >=9 else False
#         # )
#         tokenizer = AutoTokenizer.from_pretrained(args.base_model)
#         model = AutoModelForCausalLM.from_pretrained(
#             args.base_model,
#             low_cpu_mem_usage=True if torch_version >=9 else False
#         )
        
#     elif args.model_type == 'pruneLLM':
#         pruned_dict = torch.load(args.ckpt, map_location='cpu')
#         tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
#     else:
#         raise NotImplementedError

#     def input_constructor(x):
#         return {'input_ids': torch.ones(x).long().to(device)}

#     if device == "cuda":
#         model.half()
#         model = model.cuda()
    
#         with torch.cuda.device(0):
#             macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
#                                                     input_constructor = input_constructor,
#                                                     print_per_layer_stat=True, verbose=True,
#                                                     custom_modules_hooks={
#                                                         LlamaAttention: LlamaAttention_counter_hook,
#                                                         LlamaRMSNorm: rmsnorm_flops_counter_hook,
#                                                         torch.nn.SiLU: pool_flops_counter_hook,
#                                                     },)
#     else:
#         model.float()
#         macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
#                                                     input_constructor = input_constructor,
#                                                     print_per_layer_stat=True, verbose=True,
#                                                     custom_modules_hooks={
#                                                         LlamaAttention: LlamaAttention_counter_hook,
#                                                         LlamaRMSNorm: rmsnorm_flops_counter_hook,
#                                                         torch.nn.SiLU: pool_flops_counter_hook,
#                                                     },)

#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#     print("GPU Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

#     parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
#     parser.add_argument('--model_type', type=str, required=True, help = 'choose from [pretrain, pruneLLM]')
#     parser.add_argument('--ckpt', type=str, default=None)
#     parser.add_argument('--lora_ckpt', type=str, default=None)
    
#     args = parser.parse_args()
#     main(args)
