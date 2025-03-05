import os
import gc
import sys
import pathlib
import copy
import random
import argparse

from datasets import load_dataset
import torch
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

sys.path.append(
    str((pathlib.Path(__file__).parent.parent / "submodules" / "LLM_Pruner").resolve())
)

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def main(args):
    set_random_seed(args.seed)

    # Create save directory if it doesn't exist
    save_dir = os.path.join("models/pruned", args.save_ckpt_log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir="models/pruned",
        setup_sublogger=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    if args.test_before_train:
        logger.log(
            "\n==================Generation Results before Pruning================\n"
        )
        model = model.to(args.eval_device)
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                    args.eval_device
                )

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )

                result = tokenizer.decode(generation_output[0])
                logger.log(result)

        ppl = PPLMetric(
            model,
            tokenizer,
            ["wikitext2", "ptb"],
            args.max_seq_len,
            device=args.eval_device,
        )
        logger.log("PPL before pruning: {}".format(ppl))

    model.to(args.device)

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ["random", "l2", "l1", "taylor"]

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    forward_prompts = torch.tensor(
        [
            [1, 306, 4658, 278, 6593, 310, 2834, 338],
            [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
        ]
    ).to(
        args.device
    )  # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == "random":
        imp = tp.importance.RandomImportance()
    elif pruner_type == "l1":
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == "l2":
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == "taylor":
        imp = llama_pruner.TaylorImportance(
            group_reduction=args.grouping_strategy, taylor=args.taylor
        )
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))

    norm = None
    norm_pruner = None
    if "Llama" in args.base_model or "SmolLM" in args.base_model:
        norm = transformers.models.llama.modeling_llama.LlamaRMSNorm
        norm_pruner = llama_pruner.hf_rmsnorm_pruner
    elif "Qwen" in args.base_model:
        norm = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
        norm_pruner = llama_pruner.hf_rmsnorm_pruner
    elif "Mistral" in args.base_model or "Minitron" in args.base_model:
        norm = transformers.models.mistral.modeling_mistral.MistralRMSNorm
        norm_pruner = llama_pruner.hf_rmsnorm_pruner
    else:
        raise NotImplementedError(f"{args.base_model} is not supported")

    if args.block_wise or args.channel_wise:
        mlp_pruning_ratio = (
            args.mlp_pruning_ratio
            if args.mlp_pruning_ratio is not None
            else args.pruning_ratio
        )
        ch_sparsity_dict = (
            {
                model.model.layers[i].mlp.gate_proj: mlp_pruning_ratio
                for i in range(args.block_mlp_layer_start, model.config.num_hidden_layers) #args.block_mlp_layer_end
            }
            if args.block_wise
            else {}
        )
        consecutive_groups = (
            {
                layer.self_attn.k_proj: layer.self_attn.head_dim
                for layer in model.model.layers
            }
            if args.block_wise
            else {}
        )
        root_module_types = [norm] if args.channel_wise else None
        root_instances = (
            [
                model.model.layers[i].self_attn.k_proj
                for i in range(
                    args.block_attention_layer_start, model.config.num_hidden_layers #args.block_attention_layer_end
                )
            ]
            + [
                model.model.layers[i].mlp.gate_proj
                for i in range(args.block_mlp_layer_start, model.config.num_hidden_layers) #args.block_mlp_layer_end
            ]
            if args.block_wise
            else None
        )

        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ch_sparsity_dict": ch_sparsity_dict,
            "ignored_layers": [],
            "channel_groups": {},
            "consecutive_groups": consecutive_groups,
            "customized_pruners": {
                norm: norm_pruner,
            },
            "root_module_types": root_module_types,
            "root_instances": root_instances,
        }
        logger.log(
            "Pruning Attention Layer = {}".format(
                list(
                    range(
                        args.block_attention_layer_start, model.config.num_hidden_layers #args.block_attention_layer_end
                    )
                )
            )
        )
        logger.log(
            "Pruning MLP Layer = {}".format(
                list(range(args.block_mlp_layer_start, model.config.num_hidden_layers)) #args.block_mlp_layer_end
            )
        )

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ["taylor"]:
                """
                example_prompts = get_wikitext(
                    tokenizer, args.num_examples, seq_len=args.pruner_seq_len
                ).to(args.device)
                """
                example_prompts = get_examples(
                    "bookcorpus", tokenizer, args.num_examples, seq_len=args.pruner_seq_len
                ).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ["param_mix", "param_second"]:
                    for j in range(args.num_examples):
                        print(j)
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = (
                                module_param.grad
                                * module_param.grad
                                / args.num_examples
                            )
                            if hasattr(module_param, "acc_grad"):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad

                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.log(
                "After Iter {}/{}, #parameters: {}".format(
                    i + 1, args.iterative_steps, after_pruning_parameters
                )
            )

            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = (
                    layer.self_attn.q_proj.weight.data.shape[0]
                    // layer.self_attn.head_dim
                )
                layer.self_attn.num_key_value_heads = (
                    layer.self_attn.k_proj.weight.data.shape[0]
                    // layer.self_attn.head_dim
                )

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if "weight" in name:
                module.grad = None

        del pruner

        # Update configs
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.config.intermediate_size = model.model.layers[
            0
        ].mlp.gate_proj.out_features
        model.config.num_attention_heads = (
            model.model.layers[0].self_attn.q_proj.out_features
            // model.model.layers[0].self_attn.head_dim
        )
        model.config.num_key_value_heads = (
            model.model.layers[0].self_attn.k_proj.out_features
            // model.model.layers[0].self_attn.head_dim
        )

        # Update model variables
        if "Qwen2" in args.base_model:
            # Update hidden size
            for layer in model.model.layers:
                layer.self_attn.hidden_size = model.config.hidden_size

    if args.layer_wise:
        model.model.layers = model.model.layers[: args.layer]
        after_pruning_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    """
    else:
        raise NotImplementedError
    """
    logger.log(
        "#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(
            before_pruning_parameters,
            after_pruning_parameters,
            100.0 * after_pruning_parameters / before_pruning_parameters,
        )
    )

    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        if args.save_with_tokenizer:
            torch.save(
                {
                    "model": model,
                    "tokenizer": tokenizer,
                },
                logger.best_checkpoint_path,
            )
        else:
            torch.save(model, logger.best_checkpoint_path)

        model.save_pretrained(logger.log_dir)

    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log(
            "\n==================Generation Results After Pruning================\n"
        )

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                    args.eval_device
                )

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )

                result = tokenizer.decode(generation_output[0])
                logger.log(result)

        logger.log("\n==================Finish================\n")

    ppl = PPLMetric(
        model,
        tokenizer,
        ["wikitext2", "ptb"],
        args.max_seq_len,
        device=args.eval_device,
    )
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log(
        "Memory Requirement: {} MiB\n".format(
            torch.cuda.memory_allocated() / 1024 / 1024
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruning LLaMA (huggingface version)")

    # argument for parsing
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="base model name",
    )
    parser.add_argument(
        "--save_ckpt_log_name",
        type=str,
        default="llama_prune",
        help="the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.5, help="pruning ratio"
    )
    parser.add_argument(
        "--mlp_pruning_ratio", type=float, help="MLP layer pruning ratio"
    )
    parser.add_argument("--pruner_type", type=str, default="l2", help="pruner type")
    parser.add_argument("--pruner_seq_len", type=int, default=64, help="pruner type")

    # argument for generation
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="top p")
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="max sequence length"
    )

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument("--channel_wise", action="store_true", help="channel wise")
    parser.add_argument("--block_wise", action="store_true", help="block wise")
    parser.add_argument("--layer_wise", action="store_true", help="layer wise")
    parser.add_argument(
        "--layer", type=int, default=12, help="remain the previous n layers"
    )

    parser.add_argument(
        "--block_attention_layer_start",
        type=int,
        help="start layer of block attention layers",
        default=0,
    )
    parser.add_argument(
        "--block_attention_layer_end",
        type=int,
        help="end layer of block attention layers",
        default=31,
    )
    parser.add_argument(
        "--block_mlp_layer_start",
        type=int,
        help="start layer of block mlp layers",
        default=0,
    )
    parser.add_argument(
        "--block_mlp_layer_end",
        type=int,
        help="end layer of block mlp layers",
        default=31,
    )

    parser.add_argument(
        "--iterative_steps",
        type=int,
        default=1,
        help="Iteration step for pruning. Default=1",
    )
    parser.add_argument(
        "--grouping_strategy",
        type=str,
        default="sum",
        help="Reduce method for grouping",
    )
    parser.add_argument(
        "--global_pruning", action="store_true", help="whether global pruning"
    )
    parser.add_argument(
        "--taylor",
        type=str,
        default="param_first",
        help="choose from [vectorize, param_second, param_first, param_mix]",
    )
    parser.add_argument("--num_examples", type=int, default=10)

    # general argument
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--test_before_train", action="store_true", help="whether test before train"
    )
    parser.add_argument("--eval_device", type=str, default="cuda", help="eval device")
    parser.add_argument(
        "--test_after_train", action="store_true", help="whether test after train"
    )

    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--save_model", action="store_true", help="if save model")
    parser.add_argument(
        "--save_with_tokenizer", action="store_true", help="if save tokenizer"
    )
    args = parser.parse_args()

    torch_version = float(".".join(torch.__version__.split(".")[:2]))
    args.torch_version = torch_version
    main(args)
