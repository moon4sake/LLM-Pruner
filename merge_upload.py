from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from huggingface_hub import login
import os
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hf_token", type=str, default="")

    return parser.parse_args()

def main():
    args = get_args()
    login(args.hf_token)

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")

    pruned_dict = torch.load(args.base_model_name_or_path, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

    # base_model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model_name_or_path,
    #     return_dict=True,
    #     torch_dtype=torch.float16,
    #     **device_arg
    # )

    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(model, args.peft_model_path, torch_dtype=torch.float16)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    # tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    # model.save_pretrained(f"{args.output_dir}")
    # tokenizer.save_pretrained(f"{args.output_dir}")
    # print(f"Model saved to {args.output_dir}")

    REPO_NAME = args.repo_path
    AUTH_TOKEN = args.hf_token
    
    print('\n####################################################################################\n')

    ## Upload to Huggingface Hub
    print('Preparing for push to hub')
    model.push_to_hub(
        REPO_NAME, 
        use_temp_dir=True, 
        token=AUTH_TOKEN
    )
    tokenizer.push_to_hub(
        REPO_NAME, 
        use_temp_dir=True, 
        token=AUTH_TOKEN
    )
    print('\n####################################################################################\n')
    print(f'model was uploaded to {REPO_NAME}!!')


if __name__ == "__main__" :
    main()