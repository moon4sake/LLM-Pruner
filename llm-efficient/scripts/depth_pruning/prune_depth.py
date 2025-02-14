from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import os, json
import argparse
from copy import deepcopy


def check_membership(skip_list, param_name):

    for l_idx in skip_list:
        if str(l_idx) in param_name:
            return True

    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_model_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="original model path",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="./config_files/Llama-3.2-1B-Instruct_config.json",
        help="config filepath",
    )
    parser.add_argument(
        "--skip_file_path",
        type=str,
        default="./pruned_layers/remove_Llama-3.2-1B-Instruct_0.1.json",
        help="skip filepath",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="Llama-3.2-1B-Instruct_0.1",
        help="Model save path",
    )

    args = parser.parse_args()

    with open(args.skip_file_path, "r") as f:
        skip_list = json.load(f)["remove_layers"]

    config = AutoConfig.from_pretrained(args.config_file_path)
    target_layer_list = [
        i for i in range(config.num_hidden_layers) if i not in skip_list
    ]
    config.num_hidden_layers = config.num_hidden_layers - len(skip_list)

    target_model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=config.torch_dtype,
    )
    orig_model = AutoModelForCausalLM.from_pretrained(
        args.original_model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    target_model.to(orig_model.device)

    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path)

    adjusted_state_dict = {}
    for k, v in orig_model.state_dict().items():
        new_key = k

        if "model.layers." in k:
            layer_num = int(k.split(".")[2])
            if layer_num in target_layer_list:
                new_layer_num = target_layer_list.index(layer_num)
                new_key = k.replace(
                    f"model.layers.{layer_num}", f"model.layers.{new_layer_num}"
                )

                adjusted_state_dict[new_key] = v

        else:
            adjusted_state_dict[k] = v

    for param_name in target_model.state_dict().keys():
        param_value = adjusted_state_dict[param_name]
        target_model.state_dict()[param_name].copy_(param_value)

    new_config = target_model.config

    if "_name_or_path" in new_config.__dict__:
        del new_config.__dict__["_name_or_path"]

    target_model.save_pretrained(args.model_save_path)
    new_config.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
