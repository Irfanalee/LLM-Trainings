"""
DevOps Incident Responder - Fine-tuning Script
Model: nvidia/Mistral-NeMo-Minitron-8B-Instruct
Method: QLoRA (4-bit quantization + LoRA adapters)
Hardware: NVIDIA RTX A4000 16GB
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_NAME = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
MAX_SEQ_LENGTH = 1536

# LoRA config
LORA_R = 32              # Rank - capacity of adapters
LORA_ALPHA = 64          # Scaling factor (2x rank is common)
LORA_DROPOUT = 0

# Training
TRAIN_FILE = "data/processed/train_with_synthetic.jsonl"
EVAL_FILE = "data/processed/eval.jsonl"
OUTPUT_DIR = "./output"
NUM_EPOCHS = 2           # More than 2 risks overfitting
BATCH_SIZE = 1           # Per device (limited by 16GB VRAM)
GRADIENT_ACCUMULATION = 16 # Effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS = 200
EVAL_STEPS = 25

# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    print("=" * 60)
    print("DEVOPS INCIDENT RESPONDER - FINE-TUNING")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    
    print("\nLoading model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # QLoRA
    )
    
    print("Model loaded. Adding LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer

# =============================================================================
# LOAD DATASET
# =============================================================================

def load_data():
    print("\nLoading dataset...")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "eval": EVAL_FILE,
        }
    )
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Eval examples: {len(dataset['eval'])}")
    
    return dataset

# =============================================================================
# FORMAT FOR TRAINING
# =============================================================================

def format_example(example, tokenizer):
    """Format example into chat template."""
    messages = example.get("messages", [])
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def prepare_dataset(dataset, tokenizer):
    print("\nFormatting dataset...")
    
    formatted = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset["train"].column_names
    )
    
    # Show sample
    print("\n--- Sample formatted example ---")
    print(formatted["train"][0]["text"][:800])
    print("...\n")
    
    return formatted

# =============================================================================
# TRAINING
# =============================================================================

def train(model, tokenizer, dataset):
    print("Setting up trainer...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=3,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # Pack multiple examples per sequence
    )
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Total steps: ~{len(dataset['train']) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")
    
    # GPU memory info
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"\nGPU Memory: {gpu_used:.2f}GB / {gpu_mem:.2f}GB")
    
    # Train
    trainer.train()
    
    return trainer

# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(model, tokenizer):
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    
    # Save LoRA adapters
    lora_path = f"{OUTPUT_DIR}/lora_adapters"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA adapters saved to: {lora_path}")
    
    # Merge and save full model
    print("\nMerging and saving full model...")
    merged_path = f"{OUTPUT_DIR}/merged_model"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to: {merged_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load model
    model, tokenizer = load_model()
    
    # Load and prepare data
    dataset = load_data()
    formatted_dataset = prepare_dataset(dataset, tokenizer)
    
    # Train
    trainer = train(model, tokenizer, formatted_dataset)
    
    # Save
    save_model(model, tokenizer)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Test: python test_model.py")
    print("2. Quantize: python quantize.py (for CPU deployment)")


if __name__ == "__main__":
    main()
