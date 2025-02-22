"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, Any

alpaca_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name or template_name == 'alpaca':
            self.template = alpaca_template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class ZeroPrompter(object):
    __slots__ = ("_verbose")

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        
        if self._verbose:
            print(
                f"Without using prompt template!"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if instruction[-1] == '.':
            instruction = instruction[:-1] + ':'
        if instruction[-1] not in ['.', ':', '?', '!']:
            instruction = instruction + ':'
        instruction += ' '

        if input:
            if input[-1] not in ['.', ':', '?', '!']:
                input = input + '.'
            res = instruction + input
        else:
            res = instruction
        if label:
            res = f"{res} {label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.strip()

class Chat_Prompter(object):
    def __init__(self , tokenizer: Any, verbose: bool = False):
        self._verbose = verbose
        self.tokenizer = tokenizer
        self.flag= True
        
        
        self.templates=None
        
        if not hasattr(self.tokenizer, "apply_chat_template"):
            self.flag= False
            print("Tokenizer does not support `apply_chat_template`.")
            
            template_lst = {
                "alpaca": {
                    "description": "Alpaca-style chat template",
                    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n"
                },
                "chatml": {
                    "description": "ChatML-style template (OpenAI style)",
                    "prompt_input": "<|system|> You are a helpful assistant. <|user|>\n{instruction} {input} <|assistant|>\n",
                    "prompt_no_input": "<|system|> You are a helpful assistant. <|user|>\n{instruction} <|assistant|>\n"
                },
                "custom": {
                    "description": "Custom chat template",
                    "prompt_input": "[USER]: {instruction}\n[INPUT]: {input}\n[ASSISTANT]: ",
                    "prompt_no_input": "[USER]: {instruction}\n[ASSISTANT]: "
                }
            }
            
            self.templates=template_lst['custom']

    def generate_prompt(
        self,
        question: str,
        answer: Union[None, str] = None,
    ) -> str:
        
        if self.flag:
            messages = [{"role": "user", "content": question}]
            
            if answer:
                messages.append({"role": "assistant", "content": answer})
                res = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                res = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
        else:
            res = self.template["prompt_no_input"].format(question=question, answer=answer)
            
            if answer:
                res += answer
            
        if self._verbose:
            print(res)
        
        return res