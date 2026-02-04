"""
Fine-tune Qwen2.5-Coder-7B for code review using QLoRA.
Optimized for RTX A4000 16GB.
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 2048

# LoRA config
LORA_R = 32  # Rank - higher = more capacity, more VRAM
LORA_ALPHA = 64  # Scaling factor, typically 2x rank
LORA_DROPOUT = 0.05

# Training
OUTPUT_DIR = "./output"
BATCH_SIZE = 2  # Per device
GRADIENT_ACCUMULATION = 8  # Effective batch = 2 * 8 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
SAVE_STEPS = 200
LOGGING_STEPS = 25

# Data
TRAIN_FILE = "data/processed/train.jsonl"
EVAL_FILE = "data/processed/eval.jsonl"


def format_chat(example):
    """Convert our ChatML format to the model's expected format."""
    messages = example["messages"]
    
    # Qwen uses ChatML format natively
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return {"text": text}


def main():
    print("=" * 60)
    print("CODE REVIEW CRITIC - FINE-TUNING")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # -------------------------------------------------------------------------
    # Load model with QLoRA
    # -------------------------------------------------------------------------
    print("Loading model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (will use bfloat16 on Ampere)
        load_in_4bit=True,  # QLoRA
    )

    print(f"Model loaded. Adding LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Saves VRAM
        random_state=42,
    )

    # -------------------------------------------------------------------------
    # Load and prepare dataset
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "eval": EVAL_FILE}
    )
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Eval examples: {len(dataset['eval'])}")
    
    # Format to model's chat template
    print("Formatting dataset...")
    dataset = dataset.map(format_chat, remove_columns=dataset["train"].column_names)
    
    # Preview one example
    print("\n--- Sample formatted example ---")
    print(dataset["train"][0]["text"][:500])
    print("...\n")

    # -------------------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Set to "wandb" if you want logging
        seed=42,
    )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    print("Setting up trainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,  # Pack short examples together for efficiency
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Total steps: ~{len(dataset['train']) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")
    print()

    # Show GPU memory before training
    gpu_stats = torch.cuda.get_device_properties(0)
    used_memory = torch.cuda.memory_allocated() / 1024**3
    max_memory = gpu_stats.total_memory / 1024**3
    print(f"GPU Memory: {used_memory:.2f}GB / {max_memory:.2f}GB")
    print()

    trainer.train()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save LoRA adapters
    lora_path = f"{OUTPUT_DIR}/lora_adapters"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA adapters saved to: {lora_path}")
    
    # Save merged model (for easier inference)
    print("\nMerging and saving full model...")
    merged_path = f"{OUTPUT_DIR}/merged_model"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to: {merged_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Test: python test_model.py")
    print(f"2. Quantize: python quantize.py")


if __name__ == "__main__":
    main()
