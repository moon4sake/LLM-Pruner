import pandas as pd
import os
import re
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch
from typing import List



def generate_prompt(query_text, tokenizer):
    query_instruct=("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{query_text}\n\nLet's think step by step."
    )
    messages = [
        {"role": "user", "content": query_instruct}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


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
    
    class StopOnStrTokens(StoppingCriteria):
        def __init__(self, stop_strings, tokenizer):
            super().__init__()
            self.stop_strings = stop_strings
            self.tokenizer = tokenizer

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            # Decode the entire sequence (or at least the tail part) each step to check for the stop strings
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            for stop_str in self.stop_strings:
                if stop_str in text:
                    return True
            return False

    #################################################################
    # 1) Initial generation
    #################################################################
    
    # Convert prompts to token IDs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Define stopping criteria
    stop_tokens = ['<|im_end|>', '<|eot_id|>']
    stop_criteria = StoppingCriteriaList([StopOnStrTokens(stop_tokens, tokenizer)])
    
    print("Initial generation started!")
    start_time = time.time()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,         
        top_p=1.0,
        max_new_tokens=32768,    
        stopping_criteria=stop_criteria
    )
    end_time = time.time()
    print(f"Initial generation finished! Elapsed time: {end_time - start_time:.2f} seconds.\n")
    first_infer_result = [
        tokenizer.decode(output, skip_special_tokens=False)
        for output in outputs
    ]

    #################################################################
    # 2) Next round
    #################################################################

    stop_strings = [
        '<|im_start|><|im_end|>',
        '<|start_header_id|><|end_header_id|><|eot_id|>'
    ]
    
    if args.architecture_name in ['qwen', 'r1']:
        prompts = [prompt + '<|im_start|>think' for prompt in prompts]
    elif args.architecture_name == 'llama':
        prompts = [prompt + '<|start_header_id|>think' for prompt in prompts]
    else:
        raise NotImplementedError
    
    stopping_criteria = StoppingCriteriaList([
        StopOnStrTokens(stop_strings, tokenizer)
    ])

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    print("Second generation started!")
    start_time = time.time()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,          
        top_p=1.0,
        max_new_tokens=32768,      
        stopping_criteria=stopping_criteria
    )
    end_time = time.time()
    print(f"Second generation finished! Elapsed time: {end_time - start_time:.2f} seconds.\n")

    infer_result = [
        tokenizer.decode(output, skip_special_tokens=False)
        for output in outputs
    ]

    #################################################################
    # 3) Budget Forcing
    #################################################################
    
    print("Budget forcing phase started!")
    # We assume `prompts` and `infer_result` are the same length
    # and we have an existing `StopOnStrTokens` for halting on stop_tokens.
    ignore_str = "Wait"
    start_time = time.time()

    for _ in range(1):  
        prompts = [prompt + result + ignore_str for prompt, result in zip(prompts, infer_result)]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,           # temperature=0 => greedy decoding
            top_p=1.0,
            max_new_tokens=32768,      # analogous to your VLLM "max_tokens=32768"
            min_new_tokens=1,          # like min_tokens=1
            stopping_criteria=stopping_criteria
        )

        infer_result = [
            tokenizer.decode(output, skip_special_tokens=False)
            for output in outputs
        ]
    end_time = time.time()
    print(f"Budget forcing finished! Elapsed time: {end_time - start_time:.2f} seconds.\n")
    
    #################################################################
    # 4) Final process
    #################################################################
    
    print("Final process started!")
    start_time = time.time()
    
    prompts = [prompt + result + "Final Answer: " for prompt, result in zip(prompts, infer_result)]

    stop_tokens = ['<|im_end|>', '<|eot_id|>']
    stopping_criteria = StoppingCriteriaList([
        StopOnStrTokens(stop_tokens, tokenizer)
    ])

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        top_p=1.0,
        max_new_tokens=32768,
        stopping_criteria=stopping_criteria
    )
    end_time = time.time()
    print(f"Final process finished! Elapsed time: {end_time - start_time:.2f} seconds.\n")


    final_infer = [
        tokenizer.decode(output, skip_special_tokens=False)
        for output in outputs
    ]

    final_res = [prompt + result for prompt, result in zip(prompts, final_infer)]

    # Print example
    print("=" * 80)
    print("With budget forcing:\n")
    print(final_res[0])
    print("=" * 80)

    return final_res, first_infer_result


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
    
    pruned_dict = torch.load(args.model, map_location='cpu', weights_only=False)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    tokenizer.padding_side = "left"
    
    model_with_lora = PeftModel.from_pretrained(
        model,
        args.adapter,
        torch_dtype=torch.float16
    )
    model_with_lora = model_with_lora.merge_and_unload()  
    model_with_lora.to('cuda')
    
    prompts = [generate_prompt(x, tokenizer) for x in test_data_df['query'].tolist()]
    s1_result, first_result = s1_decoding(model_with_lora, tokenizer, prompts, args)
    
    df = test_result(test_data_df, s1_result, first_result)
    df.to_csv(f'{args.outdir}.csv', index=False)
    
    print('=========================== Data was saved ===========================')
    print('finished...')


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")  # model path
    parser.add_argument("--architecture_name", type=str, help="Path to the model checkpoint") 
    parser.add_argument("--adapter", type=str, help="Path to the adapter checkpoint")  # adapter path
    parser.add_argument("--data_file", type=str, default='./data/gsm8k_test')  # data file path
    parser.add_argument("--tensor_parallel_size", type=int, default=4)  # tensor parallel size
    parser.add_argument("--outdir", type=str, default='output_answer')  # output directory
    return parser.parse_args()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parse_args()
    main(args)