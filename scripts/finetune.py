import os
import sys
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)

# If you have a custom SFTTrainer, import it:
# from LLMPruner.trainer_sft import SFTTrainer
# from LLMPruner.utils.prompter import Chat_Prompter

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # -----------------------------------------------------
    # 1. Override certain arguments to match your SFT defaults:
    #    (You can conditionally override them if you wish,
    #    but here we assign them directly.)
    # -----------------------------------------------------
    args.num_epochs = 1
    args.learning_rate = 2e-4
    # We'll treat micro_batch_size as the per-device batch size:
    args.micro_batch_size = 1
    # The snippet sets gradient_accumulation_steps=4, so let's keep that in code:
    gradient_accumulation_steps = 4
    max_steps = 5000
    # The snippet also uses 256 max_seq_length by default, so we can keep
    # args.cutoff_len as-is, or force it to 256 if you prefer:
    # args.cutoff_len = 256

    # -----------------------------------------------------
    # 2. Load pruned model & tokenizer
    # -----------------------------------------------------
    pruned_dict = torch.load(args.prune_model, map_location="cpu")
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']

    # Move model to half if on GPU
    if device == "cuda":
        model.half()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Optional: If you have a prompter
    # prompter = Chat_Prompter(tokenizer)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        # Example: if data_point has 'problem' & 'solution', use them
        # For simplicity, assume there's a 'text' field
        full_prompt = data_point.get("text", "")
        tokenized_full_prompt = tokenize(full_prompt)

        if not args.train_on_inputs:
            # If you'd like to mask out the user input part,
            # implement that logic here
            pass

        return tokenized_full_prompt

    # -----------------------------------------------------
    # 3. Load Dataset
    # -----------------------------------------------------
    data = load_dataset(args.data_path)

    # If you have a train/test split, handle accordingly. Example:
    if "train" in data:
        train_data = data["train"].map(generate_and_tokenize_prompt)
        # If there's a test or validation split
        if "test" in data:
            val_data = data["test"].map(generate_and_tokenize_prompt)
        else:
            # If you want to do your own split
            train_val = data["train"].train_test_split(
                test_size=args.val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].map(generate_and_tokenize_prompt)
            val_data = train_val["test"].map(generate_and_tokenize_prompt)
    else:
        # If there's only one "train" or a single dataset
        train_data = data.map(generate_and_tokenize_prompt)
        val_data = None

    # (Optional) If you have extra validation sets, handle them here
    # e.g., if args.extra_val_dataset, load & tokenize

    # -----------------------------------------------------
    # 4. Build TrainingArguments with SFT defaults
    # -----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=f"models/finetuned/{args.out_dir}",
        # SFT snippet defaults:
        num_train_epochs=args.num_epochs,  # forcibly set to 1
        max_steps=max_steps,               # 5000
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        warmup_steps=0.03,
        learning_rate=args.learning_rate,  # 2e-4
        fp16=True,
        logging_steps=100,
        push_to_hub=False,
        report_to="tensorboard",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    # -----------------------------------------------------
    # 5. Create Trainer (or SFTTrainer)
    # -----------------------------------------------------
    # If you have a custom SFTTrainer that needs peft_config or other params:
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,  # if available
        dataset_text_field="text",  # or whichever field you're training on
        max_seq_length=args.cutoff_len,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    """

    # For demonstration, we'll use standard Hugging Face Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )

    # -----------------------------------------------------
    # 6. Train
    # -----------------------------------------------------
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # -----------------------------------------------------
    # 7. Save final model & tokenizer
    # -----------------------------------------------------
    save_path = f"models/finetuned/{args.out_dir}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full Fine-Tuning Script')

    # Model Type & Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')  # Not used directly in new script, but kept
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=50, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # LLM hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true",
                        help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true",
                        help="faster, but produces an odd training loss curve")

    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    # New argument: which subdirectory under models/finetuned/
    parser.add_argument('--out_dir', type=str, default="llama3",
                        help="Folder under models/finetuned where we save final model")

    args = parser.parse_args()
    main(args)
