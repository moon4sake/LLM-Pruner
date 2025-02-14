from llm_efficient.depth_pruning.skip_metric import block_influence
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import argparse
import json, os


def load_wikitext():
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [
        text
        for text in data["text"]
        if text.strip() != "" and len(text.split(" ")) > 20
    ]

    # Decode the output tokens to text
    output_texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
    ]

    return output_texts


# python compute_important_score.py --n_prune_layers 16
# python compute_important_score.py --n_prune_layers 9


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="base model"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="layer_score_results",
        help="layer score result",
    )

    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="Model Max Length"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="max new tokens"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="max new tokens")

    parser.add_argument("--percent", type=float, default=0.1, help="percent")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="auto", device_map="auto", use_cache=False
    )
    model.gradient_checkpointing_enable()

    n_prune_layers = int(model.config.num_hidden_layers * args.percent)
    save_path = f"./{args.output_folder}/remove_{args.percent}.json"

    angular = False
    importances = [
        0 for _ in range(model.config.num_hidden_layers)
    ]  # layer-wise importance scores

    print("###################")
    print("# Number of Layer: ", model.config.num_hidden_layers)
    print("###################")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        trust_remote_code=True,
        # padding_side="right",
        # use_fast=False,
    )

    if tokenizer.pad_token != tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    ##########################################################################################

    calibration_dataset = load_wikitext()

    seed_dataloader = DataLoader(calibration_dataset, batch_size=args.batch_size)

    ##########################################################################################

    model.eval()

    for b_idx, input_text in enumerate(tqdm(seed_dataloader)):

        with torch.no_grad():
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=(args.model_max_length - args.max_new_tokens),
            )

            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # import pdb

            # pdb.set_trace()

            hidden_states = outputs["hidden_states"][-1]

        n = 1
        if angular:
            n = n_prune_layers

        for i in range(len(hidden_states) - n):
            in_hidden = hidden_states[i]
            out_hidden = hidden_states[i + n]
            if angular:
                in_hidden = in_hidden[:, -1:]
                out_hidden = out_hidden[:, -1:]

            importances[i] += (
                block_influence(in_hidden, out_hidden, angular=angular)
                .sum()
                .cpu()
                .item()
            )

    layers_to_remove = []

    if angular:
        start_layer = np.argsort(np.array(importances[: -n_prune_layers + 1]))[0]
        layers_to_remove = list(range(start_layer, start_layer + n_prune_layers))

    elif not layers_to_remove and n_prune_layers:
        layers_to_remove = np.argsort(np.array(importances))[:n_prune_layers].tolist()

    layers_to_remove.sort()
    skip_list = {"remove_layers": layers_to_remove}

    with open(save_path, "w") as f:
        json.dump(skip_list, f, indent=2, ensure_ascii=False)
