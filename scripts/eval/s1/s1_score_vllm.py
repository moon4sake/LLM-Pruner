import pandas as pd
import os
import re
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.outputs import RequestOutput
from typing import List


def generate_prompt(query_text,tokenizer):
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
    # 1. "Final Answer:" 패턴에서 숫자 추출 (우선순위 가장 높음)
    final_answer_match = re.search(r'Final Answer:\s*(\\boxed\{[^\d]*(\d+)[^\d]*\}|\d{1,3}(?:[,\.]\d{3})*)', output_text)

    if final_answer_match:
        final_answer = final_answer_match.group(2) if final_answer_match.group(2) else final_answer_match.group(1)
        return int(final_answer.replace(',', '').replace('.', ''))  # 천 단위 구분자 제거 후 변환

    # 2. "The answer is:" 패턴에서 숫자 추출 (대체 역할)
    the_answer_match = re.search(r'The answer is:\s*(\\boxed\{[^\d]*(\d+)[^\d]*\}|\d{1,3}(?:[,\.]\d{3})*)', output_text)

    if the_answer_match:
        the_answer = the_answer_match.group(2) if the_answer_match.group(2) else the_answer_match.group(1)
        return int(the_answer.replace(',', '').replace('.', ''))  # 천 단위 구분자 제거 후 변환

    # 3. 일반 숫자 추출 (천 단위 구분자 포함)
    numbers = re.findall(r'\b\d{1,3}(?:[,\.]\d{3})*\b', output_text)

    # 4. \boxed{} 안의 숫자 추출
    boxed_numbers = re.findall(r'\\boxed\{[^\d]*(\d+)[^\d]*\}', output_text)

    # 5. 모든 숫자를 하나의 리스트로 합침
    all_numbers = numbers + boxed_numbers

    if all_numbers:
        # 마지막 숫자 선택 (기존 방식)
        final_answer = all_numbers[-1]
        return int(final_answer.replace(',', '').replace('.', ''))  # 콤마 제거 후 변환

    return None
    
    
    
    
def s1_decoding(model,tokenizer,prompts,args):
    
    
    # initial generation  (no budget forcing result)
    stop_tokens=['<|im_end|>','<|eot_id|>']    
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=0,
        top_p=1.0, 
        max_tokens=32768, 
        stop=stop_tokens,
        skip_special_tokens=False
        )

    outputs: List[RequestOutput] = model.generate(prompts, sampling_params)  

    fist_infer_result= [output.outputs[0].text for output in outputs]
    
    ########################################################################################    

    stop_tokens=['<|im_start|><|im_end|>','<|start_header_id|><|end_header_id|><|eot_id|>']    
        
        
    sampling_params_for_budget = SamplingParams(
        temperature=0.0,
        min_tokens=0,
        top_p=1.0,
        max_tokens=32768,
        stop=stop_tokens,
        skip_special_tokens=False,
    )
    
    if args.architecture_name=='qwen' or args.architecture_name=='r1':
        prompts = [prompt + '<|im_start|>think' for prompt in prompts]
        
    elif args.architecture_name=='llama':
        prompts = [prompt + '<|start_header_id|>think' for prompt in prompts]
        
    else:
        pass
        
        
    outputs: List[RequestOutput] = model.generate(prompts, sampling_params) 
    infer_result= [output.outputs[0].text for output in outputs]

    
    
    ####################################################################################
    # Budget Forcing
    
    ignore_str = "Wait"
    

    for _ in range(1):
        prompts = [prompt + res+ ignore_str for prompt, res in zip(prompts, infer_result) ]
        
        sampling_params = SamplingParams(
            max_tokens=32768,
            min_tokens=1,
            stop=stop_tokens,
            skip_special_tokens=False,
            temperature=0.0,
        )
        outputs: List[RequestOutput] = model.generate(prompts, sampling_params) 
        infer_result= [output.outputs[0].text for output in outputs]
        
        
        
    ####################################################################################
    
    prompts = [prompt + res + 'Final Answer: ' for prompt, res in zip(prompts, infer_result) ]
    
    stop_tokens=['<|im_end|>','<|eot_id|>']
    
    
    sampling_params = SamplingParams(
        max_tokens=32768,
        min_tokens=0,
        stop=stop_tokens,
        skip_special_tokens=False,
        temperature=0.0,
    )
    
    
    outputs: List[RequestOutput] = model.generate(prompts, sampling_params) 
    
    final_infer= [output.outputs[0].text for output in outputs]
    
    final_res = [prompt + res for prompt, res in zip(prompts, final_infer) ]
    
    print("="*80)
    print("With budget forcing:\n")
    print(final_res[0])
    print("="*80)
    
    return final_res, fist_infer_result
    

        
def test_result(df, s1_out, origin_out):
    
    df['s1_output'] = s1_out
    df['origin_output'] = origin_out
    
    
    extracted_s1 =  list(map(extract_answer, s1_out))
    extracted_origin =  list(map(extract_answer, origin_out))
    
    df['s1_y'] = extracted_s1
    df['origin_y'] = extracted_origin    

    df['s1_y_correct'] = df['s1_y'] == df['GT']
    df['origin_y_correct'] = df['origin_y'] == df['GT']
    
    s1_accuracy = df['s1_y_correct'].mean()  
    original_accuracy = df['origin_y_correct'].mean()
    

    print('='*80)
    print(f"origin_y Accuracy: {original_accuracy:.8%}")
    print(f"s1_y Accuracy: {s1_accuracy:.8%}")
    print('='*80)
    
    
    return df
    
    



def main(args):
    # test_dataset
    test_data_df = pd.read_csv(args.data_file)
    test_data_df = test_data_df[:100]
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="left")
    
    prompts = [generate_prompt(x, tokenizer) for x in test_data_df['query'].tolist()]

    model = LLM(model=args.model,tensor_parallel_size=args.tensor_parallel_size)
        
    s1_result, first_result = s1_decoding(model,tokenizer, prompts, args)
    
    df= test_result(test_data_df, s1_result, first_result)
    
    df.to_csv(f'{args.outdir}.csv',index=False)
    
    print('=========================== Data was saved ===========================')
    print('finished...')
    

        


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--architecture_name", type=str, default='llama')  # model architecture, llama, qwen, r1 etc
    parser.add_argument("--tokenizer", type=str)  # tokenizer path
    parser.add_argument("--data_file", type=str, default='./data/gsm8k_test')  # data_path
    parser.add_argument("--tensor_parallel_size", type=int, default=4)  # tensor_parallel_size
    parser.add_argument("--outdir", type=str,default='output_answer')  # output result path name
    return parser.parse_args()


if __name__ == "__main__":
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    args = parse_args()
    main(args)