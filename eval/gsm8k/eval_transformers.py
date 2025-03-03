import os
import argparse
import json
import re
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from fraction import Fraction
import sys
import pandas as pd
from tqdm import tqdm
MAX_INT = sys.maxsize



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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model, tokenizer, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop_tokens = ['<|im_end|>', '<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', '<|im_start|>user<|im_end|>', 'Q:', '</s>']
    stop_criteria = StoppingCriteriaList([StopOnStrTokens(stop_tokens, tokenizer)])
    
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_gsm8k_ins, gsm8k_answers)), total=len(batch_gsm8k_ins), desc="Processing batches"):
        if not isinstance(prompt, list):
            prompt = [prompt]
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,         
            top_p=0.6,  
            temperature=0.95,
            max_new_tokens=4096,
            stopping_criteria=stop_criteria
        )
        
        for out_ids in outputs:
            generated_text = tokenizer.decode(out_ids, skip_special_tokens=False)
            res_completions.append(generated_text)
        
        break

    invalid_outputs = []
    gt=[]

    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        gt.append(prompt_answer)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    
    df=pd.DataFrame({'answer':res_completions, 'GT': gt})

    df.to_csv(f'{args.outdir}.csv',index=False)
    
    print('len invalid outputs ====', len(invalid_outputs), ', invalid_outputs===', invalid_outputs)
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")  # model path
    parser.add_argument("--adapter", type=str, help="Path to the adapter checkpoint")  # adapter path
    parser.add_argument("--tokenizer", type=str)  # tokenizer path
    parser.add_argument("--outdir", type=str,default='output_answer')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=100)  # batch_size
    parser.add_argument("--is_pretrained", action="store_true", default=False,
                        help="If set, indicates a Hugging Face pretrained model is used.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.is_pretrained:
        tokenizer_path = args.tokenizer if args.tokenizer else args.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16
        ).cuda()
    else:
        pruned_dict = torch.load(args.model, map_location='cpu', weights_only=False)
        tokenizer, base_model = pruned_dict['tokenizer'], pruned_dict['model']
        tokenizer.padding_side = "left"

        # If there's an adapter, merge it
        if args.adapter is not None:
            model_with_lora = PeftModel.from_pretrained(
                base_model,
                args.adapter,
                torch_dtype=torch.float16
            )
            model = model_with_lora.merge_and_unload()
        else:
            model = base_model
        
        model.to('cuda')

    gsm8k_test(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
    )