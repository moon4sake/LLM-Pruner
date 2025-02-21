
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--hf_token", type =str, default='')

    
    return parser.parse_args()



def main():

    args = get_args()

    print('\n####################################################################################\n')
    print('\narguments are received')
    # Model & Tokenizer loading

    ####################################################################################
    ## pruned_dict = torch.load(args.base_model_name_or_path, map_location='cpu')
    ## tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    ####################################################################################

    print('\n####################################################################################\n')
    print("\nmodel is loading...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    print('\n####################################################################################\n')

    print('model was loaded')
    # Repository 생성 & model upload
    REPO_NAME = args.repo_path
    AUTH_TOKEN = args.hf_token
    
    print('\n####################################################################################\n')

    ## Upload to Huggingface Hub
    print('Preparing for push to hub')
    model.push_to_hub(
        REPO_NAME, 
        use_temp_dir=True, 
        use_auth_token=AUTH_TOKEN
    )
    tokenizer.push_to_hub(
        REPO_NAME, 
        use_temp_dir=True, 
        use_auth_token=AUTH_TOKEN
    )
    print('\n####################################################################################\n')
    print(f'model was uploaded to {REPO_NAME}!!')


if __name__ == "__main__" :
    main()

