import pandas as pd
import re
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.outputs import RequestOutput
from typing import List
import torch



def generate_extract_prompt_with_question(
    extract_template: str,
    question_text: str,
    answer_text: str,
    tokenizer
) -> str:
   
    filled_prompt = extract_template.format(
        question_text=question_text.strip(),
        answer_text=answer_text.strip(),
    )
    messages = [
        {"role": "user", "content": filled_prompt}
    ]
    prompt_for_model = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_for_model += "\n- **Model's Final Answer is:** "

    return prompt_for_model




def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model 이름 그냥 적는겁니다
    parser.add_argument("--data_file", type=str, default='./')  # data_path-> 모델 answer랑 정답 적힌 df path
    parser.add_argument("--tensor_parallel_size", type=int, default=4)  # tensor_parallel_size
    return parser.parse_args()


def main():
    args = parse_args()
    df=pd.read_csv(args.data_file)
    
    
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    model = LLM(
        model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=16392,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        dtype='auto',
        enforce_eager=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left")

    template_path='/mnt/home/sy/synthetic_data_gen/Medical_test/medical_data_w/o/extract_answer.txt'
    
    with open(template_path, 'r',encoding='utf-8') as file:
        extract_template = file.read()  
        
    lst= []
    for i in range(len(df)):
        prompt=generate_extract_prompt_with_question(extract_template,df['query'][i],df['answer'][i],tokenizer)
        lst.append(prompt)
        
    df['prompts']=lst
    
    sampling_params = SamplingParams(temperature=0.9, top_p=0.7, top_k=10,repetition_penalty=1, max_tokens=100)
    
    outputs: List[RequestOutput] = model.generate(lst, sampling_params)
    
    generated_texts = [output.outputs[0].text for output in outputs]
    
    df['gen_answer'] = generated_texts
    
    count=0
    for i in range(len(df)):
        if int(generated_texts.strip().replace(',',''))==int(df['GT'][i]): # 숫자면 수정하시고 dtype int로 변경하세요
            count+=1
            
        else:
            pass
    
    print('='*80)
    print(f'ACC of {args.model}:')
    print('ACC:',count/len(df))
    print('='*80)
    
    
    
    df.to_csv(args.data_file,index=False)
    
    

if __name__ == "__main__":
    main()