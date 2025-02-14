import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from torch.utils.data import DataLoader
import json

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)

from trl import setup_chat_format, SFTTrainer
from llm_efficient.dynamic_batching.custom_trainer import CustomSFTTrainer
from copy import deepcopy
from trl.trainer.utils import ConstantLengthDataset
from trl.extras.dataset_formatting import get_formatting_func_from_dataset

from transformers import Trainer


def load_each_domain_eval_data(data_args):

    eval_datasets = []

    dataset_mixer = deepcopy(data_args.dataset_mixer)

    for dataname in dataset_mixer.keys():

        each_mixer = {}
        each_mixer[dataname] = dataset_mixer[dataname]

        data_args.dataset_mixer = each_mixer

        eval_dataset = get_datasets(
            data_args,
            splits=["test_sft"],
            configs=data_args.dataset_configs,
            columns_to_keep=[
                "messages",
                "chosen",
                "rejected",
                "prompt",
                "completion",
                "label",
            ],
        )

        eval_datasets.append(eval_dataset)

    return eval_datasets


if __name__ == "__main__":

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    set_seed(training_args.seed)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    tokenizer = get_tokenizer(model_args, data_args)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    # For ChatML we need to add special tokens and resize the embedding layer
    if (
        "<|im_start|>" in tokenizer.chat_template
        and "gemma-tokenizer-chatml" not in tokenizer.name_or_path
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        model, tokenizer = setup_chat_format(model, tokenizer)
        model_kwargs = None

    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=[
            "messages",
            "chosen",
            "rejected",
            "prompt",
            "completion",
            "label",
        ],
    )
    column_names = list(raw_datasets["train"].features)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    domain_eval_datasets = load_each_domain_eval_data(data_args)

    for idx, each_eval_dataset in enumerate(domain_eval_datasets):
        eval_columns = list(each_eval_dataset["test"].features)

        domain_eval_datasets[idx] = each_eval_dataset.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_columns,
            desc="Applying chat template",
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs,
    )

    all_results = []

    for idx, each_eval_dataset in enumerate(domain_eval_datasets):
        domain_eval_datasets[idx] = each_eval_dataset.filter(
            decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=1
        )

        target_dataset = domain_eval_datasets[idx]
        formatting_func = get_formatting_func_from_dataset(target_dataset, tokenizer)

        domain_eval_datasets[idx] = ConstantLengthDataset(
            tokenizer,
            domain_eval_datasets[idx]["test"],
            dataset_text_field=(
                None
                if formatting_func is not None
                else training_args.dataset_text_field
            ),
            formatting_func=formatting_func,
            seq_length=training_args.max_seq_length,
            infinite=False,
            num_of_sequences=training_args.num_of_sequences,
            chars_per_token=training_args.chars_per_token,
            eos_token_id=tokenizer.eos_token_id,
            append_concat_token=True,
            add_special_tokens=True,
        )

        output_metric = trainer.evaluate(
            eval_dataset=domain_eval_datasets[idx],
            ignore_keys=None,
            metric_key_prefix=f"{idx}",
        )

        all_results.append(output_metric)

    with open("target_loss.json", "w") as f:
        json.dump(all_results, f, indent=2)
