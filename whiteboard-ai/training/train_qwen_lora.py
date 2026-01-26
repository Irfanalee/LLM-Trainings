#!/usr/bin/env python3
"""
Qwen2.5-7B QLoRA Fine-tuning for Action Item Extraction
Optimized for RTX A4000 16GB GPU

Uses:
- 4-bit quantization (QLoRA)
- Gradient checkpointing
- Flash Attention 2 (if available)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer


def load_action_item_dataset(data_path: str) -> Dataset:
    """
    Load action item dataset from JSON or JSONL file

    Expected format:
    {
        "instruction": "Extract action items...",
        "input": "Meeting notes...",
        "output": "[{action_items}]"
    }
    """
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)

    return Dataset.from_list(data)


def format_prompt(example: Dict, tokenizer) -> str:
    """
    Format a single example for training using Qwen2.5 chat template
    """
    system_message = "You are a helpful assistant that extracts action items from meeting notes. Always return valid JSON."

    user_message = f"{example['instruction']}\n\n{example['input']}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example['output']}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return text


def train_qwen_lora(
    train_data: str,
    val_data: str = None,
    output_dir: str = "./runs/qwen_action_items",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.1,
    max_seq_length: int = 2048
):
    """
    Fine-tune Qwen2.5-7B with QLoRA for action item extraction

    Memory optimization for RTX A4000 16GB:
    - 4-bit quantization
    - Gradient checkpointing
    - Small batch size + gradient accumulation
    """
    print("="*60)
    print("Qwen2.5-7B QLoRA Training for Action Item Extraction")
    print("="*60)

    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: No GPU available! Training will be very slow.")
    else:
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    print(f"Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # Use "flash_attention_2" if available
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # LoRA configuration
    print(f"\nApplying LoRA (r={lora_r}, alpha={lora_alpha})")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_action_item_dataset(train_data)
    print(f"  Train: {len(train_dataset)} samples")

    val_dataset = None
    if val_data:
        val_dataset = load_action_item_dataset(val_data)
        print(f"  Val: {len(val_dataset)} samples")

    # Format function for SFTTrainer
    def formatting_func(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            example = {
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i]
            }
            text = format_prompt(example, tokenizer)
            texts.append(text)
        return texts

    # Training arguments (optimized for A4000 16GB)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # Memory optimizations
        fp16=False,  # Use bf16 instead
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,

        # Logging
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,

        # Other
        report_to="none",  # Set to "tensorboard" for logging
        push_to_hub=False,
        dataloader_num_workers=2,
        group_by_length=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        packing=False,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the LoRA adapter
    print("\nSaving LoRA adapter...")
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    print(f"\nTraining complete!")
    print(f"LoRA adapter saved to: {output_dir}/lora_adapter")

    return model, tokenizer


def merge_and_save(
    base_model: str,
    lora_adapter_path: str,
    output_path: str
):
    """
    Merge LoRA adapter with base model and save
    """
    from peft import PeftModel

    print("Merging LoRA adapter with base model...")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # Merge
    model = model.merge_and_unload()

    # Save
    model.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


def inference_test(
    lora_adapter_path: str,
    test_input: str = None
):
    """
    Test the fine-tuned model
    """
    from peft import PeftModel

    print("\nTesting fine-tuned model...")

    # Load with quantization for inference
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(lora_adapter_path), "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Test input
    if test_input is None:
        test_input = """Meeting: Sprint Planning

Discussion:
1. Reviewed Q4 budget with the team
2. Sarah presented updates on design mockups
3. Need to address security audit findings

Action Items:
- [ ] Review security audit report - @John by Friday
- TODO: Update API documentation - Sarah by next week
- Action: Schedule deployment review. Owner: Mike. Deadline: Thursday"""

    # Create prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts action items from meeting notes. Always return valid JSON."},
        {"role": "user", "content": f"Extract all action items from the following meeting notes. Return a JSON array with task, assignee, deadline, and priority for each item.\n\n{test_input}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.95
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("\nInput:")
    print(test_input)
    print("\nExtracted Action Items:")
    print(response)

    return response


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen2.5 with QLoRA')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train with QLoRA')
    train_parser.add_argument('--train-data', required=True, help='Training data (JSON/JSONL)')
    train_parser.add_argument('--val-data', help='Validation data (optional)')
    train_parser.add_argument('--output-dir', default='./runs/qwen_action_items', help='Output directory')
    train_parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct', help='Base model')
    train_parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    train_parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    train_parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    train_parser.add_argument('--lora-r', type=int, default=64, help='LoRA rank')
    train_parser.add_argument('--lora-alpha', type=int, default=128, help='LoRA alpha')
    train_parser.add_argument('--max-seq-len', type=int, default=2048, help='Max sequence length')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test fine-tuned model')
    test_parser.add_argument('--adapter', required=True, help='Path to LoRA adapter')
    test_parser.add_argument('--input', help='Test input (optional)')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge LoRA with base model')
    merge_parser.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct', help='Base model')
    merge_parser.add_argument('--adapter', required=True, help='Path to LoRA adapter')
    merge_parser.add_argument('--output', required=True, help='Output path for merged model')

    args = parser.parse_args()

    if args.command == 'train':
        train_qwen_lora(
            train_data=args.train_data,
            val_data=args.val_data,
            output_dir=args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_seq_length=args.max_seq_len
        )
    elif args.command == 'test':
        inference_test(args.adapter, args.input)
    elif args.command == 'merge':
        merge_and_save(args.base_model, args.adapter, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# Example usage:
"""
# 1. Generate synthetic training data:
python scripts/generate_synthetic_data.py --output-dir datasets/synthetic --num-action-items 5000

# 2. Prepare data for training:
python scripts/prepare_datasets.py action \\
    --input-jsonl datasets/synthetic/action_items.jsonl \\
    --output-dir datasets/action_items_prepared

# 3. Train with QLoRA:
python training/train_qwen_lora.py train \\
    --train-data datasets/action_items_prepared/train.json \\
    --val-data datasets/action_items_prepared/val.json \\
    --output-dir runs/qwen_action_items \\
    --epochs 3 \\
    --batch-size 2 \\
    --grad-accum 8

# 4. Test the model:
python training/train_qwen_lora.py test \\
    --adapter runs/qwen_action_items/lora_adapter

# Memory usage on RTX A4000:
# - Batch size 2 + grad accum 8 = effective batch 16
# - ~10-12GB VRAM usage
# - Training time: ~2-3 hours for 5000 samples, 3 epochs

# Tips for faster training:
# - Use --batch-size 4 --grad-accum 4 if you have memory headroom
# - Reduce --max-seq-len to 1024 if your meeting notes are short
# - Use --lora-r 32 --lora-alpha 64 for faster training (slightly less quality)
"""
