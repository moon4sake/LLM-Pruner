import pandas as pd
import os
import re
import argparse
from transformers import AutoTokenizer
import torch
from typing import List


def generate_prompt(query_text, tokenizer):
    query_instruct = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{query_text}\n\nLet's think step by step."
    )
    return tokenizer(query_instruct, return_tensors='pt')['input_ids']


def extract_answer(output_text):
    final_answer_match = re.search(r'Final Answer:\s*(\\boxed\{[^\d]*(\d+)[^\d]*\}|\d{1,3}(?:[,\.]\d{3})*)', output_text)
    if final_answer_match:
        final_answer = final_answer_match.group(2) if final_answer_match.group(2) else final_answer_match.group(1)
        return int(final_answer.replace(',', '').replace('.', ''))
    the_answer_match = re.search(r'The answer is:\s*(\\boxed\{[^\d]*(\d+)[^\d]*\}|\d{1,3}(?:[,\.]\d{3})*)', output_text)
    if the_answer_match:
        the_answer = the_answer_match.group(2) if the_answer_match.group(2) else the_answer_match.group(1)
        return int(the_answer.replace(',', '').replace('.', ''))
    numbers = re.findall(r'\b\d{1,3}(?:[,\.]\d{3})*\b', output_text)
    boxed_numbers = re.findall(r'\\boxed\{[^\d]*(\d+)[^\d]*\}', output_text)
    all_numbers = numbers + boxed_numbers
    if all_numbers:
        final_answer = all_numbers[-1]
        return int(final_answer.replace(',', '').replace('.', ''))
    return None


def s1_decoding(model, tokenizer, prompts, args):
    stop_tokens = tokenizer.convert_tokens_to_ids(['<|im_end|>', '<|eot_id|>'])
    outputs = []
    for prompt in prompts:
        input_ids = prompt.to('cuda')
        output_ids = model.generate(input_ids, max_length=512, eos_token_id=stop_tokens[0])
        result_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(result_text)
    return outputs, outputs


def test_result(df, s1_out, origin_out):
    df['s1_output'] = s1_out
    df['origin_output'] = origin_out
    extracted_s1 = list(map(extract_answer, s1_out))
    extracted_origin = list(map(extract_answer, origin_out))
    df['s1_y'] = extracted_s1
    df['origin_y'] = extracted_origin    
    df['s1_y_correct'] = df['s1_y'] == df['GT']
    df['origin_y_correct'] = df['origin_y'] == df['GT']
    s1_accuracy = df['s1_y_correct'].mean()  
    original_accuracy = df['origin_y_correct'].mean()
    print('=' * 80)
    print(f"origin_y Accuracy: {original_accuracy:.8%}")
    print(f"s1_y Accuracy: {s1_accuracy:.8%}")
    print('=' * 80)
    return df


def main(args):
    test_data_df = pd.read_csv(args.data_file)
    test_data_df = test_data_df[:100]

    # Load model and tokenizer from checkpoint
    pruned_dict = torch.load(args.model, map_location='cpu', weights_only=False)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    
    # Load adapter (if necessary)
    adapter_dict = torch.load(args.adapter, map_location='cpu', weights_only=False)
    model.load_state_dict(adapter_dict, strict=False)  # You might need to customize this line

    model.to('cuda')
    
    prompts = [generate_prompt(x, tokenizer) for x in test_data_df['query'].tolist()]
    s1_result, first_result = s1_decoding(model, tokenizer, prompts, args)
    
    df = test_result(test_data_df, s1_result, first_result)
    df.to_csv(f'{args.outdir}.csv', index=False)
    
    print('=========================== Data was saved ===========================')
    print('finished...')


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")  # model path
    parser.add_argument("--adapter", type=str, help="Path to the adapter checkpoint")  # adapter path
    parser.add_argument("--tokenizer", type=str)  # tokenizer path
    parser.add_argument("--data_file", type=str, default='./data/gsm8k_test')  # data file path
    parser.add_argument("--tensor_parallel_size", type=int, default=4)  # tensor parallel size
    parser.add_argument("--outdir", type=str, default='output_answer')  # output directory
    return parser.parse_args()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parse_args()
    main(args)