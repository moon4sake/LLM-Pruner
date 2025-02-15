import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from torch.utils.data import DataLoader


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

from trl import setup_chat_format
from llm_efficient.dynamic_batching.custom_trainer import CustomSFTTrainer
from copy import deepcopy

logger = logging.getLogger(__name__)


data_map = {
    "HuggingFaceH4/ultrachat_200k": 951000,
    "HuggingFaceTB/self-oss-instruct-sc2-H4": 481000,
    "HuggingFaceTB/OpenHermes-2.5-H4": 951000,
    "HuggingFaceTB/everyday-conversations-llama3.1-2k": 2260,
    "HuggingFaceTB/Magpie-Pro-300K-Filtered-H4": 270000,
}

target_loss_map = {
    "HuggingFaceH4/ultrachat_200k": 1.6346638202667236,
    "HuggingFaceTB/self-oss-instruct-sc2-H4": 0.94910728931427,
    "HuggingFaceTB/OpenHermes-2.5-H4": 1.5247459411621094,
    "HuggingFaceTB/everyday-conversations-llama3.1-2k": 1.168951392173767,
}


def load_each_domain_eval_data(data_args):

    eval_datasets, proportion, target_loss = [], [], []
    dataset_mixer = deepcopy(data_args.dataset_mixer)

    for dataname in dataset_mixer.keys():
        new_data_args = deepcopy(data_args)
        each_mixer = {}
        each_mixer[dataname] = dataset_mixer[dataname]

        new_data_args.dataset_mixer = each_mixer

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

        num_of_data = int(data_map[dataname] * dataset_mixer[dataname])

        proportion.append(num_of_data)
        target_loss.append(target_loss_map[dataname])

    total_number = sum(proportion)
    proportion = [round(element / total_number, 4) for element in proportion]

    return eval_datasets, proportion, target_loss


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############

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

    ########################################################
    domain_eval_datasets, proportion, target_loss = load_each_domain_eval_data(
        data_args
    )
    ########################################################

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
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

    #####################
    # Apply chat template
    #####################
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

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(
        decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=1
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    for idx, each_eval_dataset in enumerate(domain_eval_datasets):
        domain_eval_datasets[idx] = each_eval_dataset.filter(
            decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=1
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ### 사이즈 줄이기
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(100))

    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(
                f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}"
            )

    ########################
    # Initialize the Trainer
    ########################

    trainer = CustomSFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        data_args=data_args,
        each_domain_eval_datasets=domain_eval_datasets,
        proportion=proportion,
        target_loss=target_loss,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
