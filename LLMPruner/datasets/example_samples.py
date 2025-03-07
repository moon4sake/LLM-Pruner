import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_wikitext(tokenizer, n_samples, seq_len):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    tokenized_samples, history = [], []
    for p in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_openr1_math(tokenizer, n_samples, seq_len=4096):
    """
    Return (input_ids, labels) for a small calibration dataset,
    where the model is given a math 'problem' and the 'solution' is predicted.
    """
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")

    # We'll store multiple samples and then batch them
    input_ids_list = []
    labels_list = []
    used_indices = set()

    while len(input_ids_list) < n_samples:
        i = random.randint(0, len(dataset) - 1)
        if i in used_indices:
            continue
        used_indices.add(i)

        row = dataset[i]
        problem_text = row["problem"]
        solution_text = row["solution"]

        # Build a text prompt: "Question: <problem>\nAnswer: <solution>"
        # but we only want the model's loss to apply to the solution portion
        prompt_text = f"Question: {problem_text}\nAnswer:"  # no solution text yet
        full_text = f"{prompt_text} {solution_text}"

        # Tokenize the entire text
        encoded_full = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,  # we'll handle truncation ourselves
            add_special_tokens=True
        )
        full_ids = encoded_full["input_ids"]  # shape: [1, seq_len_full]

        # Tokenize just the prompt (problem portion)
        encoded_prompt = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        prompt_len = encoded_prompt["input_ids"].shape[1]

        # If the full text is shorter than seq_len, skip if it's too short;
        # otherwise, pick a random slice of length seq_len
        if full_ids.shape[1] < seq_len:
            continue
        start_idx = random.randint(0, full_ids.shape[1] - seq_len)

        # Slice out exactly seq_len tokens
        input_ids_slice = full_ids[:, start_idx : start_idx + seq_len]

        # Build labels: same shape as input_ids, but ignore (=-100) the prompt portion
        labels_slice = input_ids_slice.clone()
        # Figure out which part of the slice overlaps with the prompt
        prompt_end = prompt_len - start_idx  # how far into this slice the prompt might go
        prompt_end = max(min(prompt_end, seq_len), 0)  # clamp to [0, seq_len]
        # set prompt region to -100
        if prompt_end > 0:
            labels_slice[:, :prompt_end] = -100

        input_ids_list.append(input_ids_slice)
        labels_list.append(labels_slice)

    # Combine into a single batch
    batch_input_ids = torch.cat(input_ids_list, dim=0)   # [n_samples, seq_len]
    batch_labels    = torch.cat(labels_list,    dim=0)   # [n_samples, seq_len]
    return batch_input_ids, batch_labels

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'wikitext':
        return get_wikitext(tokenizer, n_samples, seq_len)
    elif dataset == 'openr1_math':
        return get_openr1_math(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
