"""
Train Document Intelligence MoE Model
Fine-tune Qwen3-30B-A3B using Unsloth's optimized MoE training

Model: Qwen3-30B-A3B (30B total, 3B active per token)
Method: QLoRA with Unsloth's MoE Triton kernels
Hardware: NVIDIA RTX A4000 16GB (or similar)

Requirements:
    pip install unsloth
    pip install --upgrade transformers trl
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_PATH = Path(__file__).parent
TRAIN_FILE = BASE_PATH / "data" / "training" / "train.jsonl"
EVAL_FILE = BASE_PATH / "data" / "training" / "eval.jsonl"
OUTPUT_DIR = BASE_PATH / "output"

# Model - Qwen3 MoE
MODEL_NAME =  "unsloth/gpt-oss-20b" #"unsloth/Qwen3-30B-A3B"  # 30B total, 3B active
MAX_SEQ_LENGTH = 1536

# LoRA Configuration
LORA_R = 32              # Rank
LORA_ALPHA = 64          # Scaling (2x rank)
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP / Experts
]

# Training Configuration
NUM_EPOCHS = 2
BATCH_SIZE = 1           # Small for 16GB VRAM
GRADIENT_ACCUMULATION = 16  # Effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS = 100
EVAL_STEPS = 100

# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    """Load Qwen3-30B-A3B with Unsloth optimizations."""
    print("=" * 60)
    print("LOADING QWEN3-30B-A3B (MoE)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Check VRAM
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    print("\nLoading model with 4-bit quantization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (bf16 for newer GPUs)
        load_in_4bit=True,  # QLoRA - required for 16GB VRAM
    )
    
    print("Adding LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=42,
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_data():
    """Load training and evaluation data."""
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "eval": str(EVAL_FILE),
        }
    )
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Eval examples: {len(dataset['eval'])}")
    
    return dataset


def format_example(example, tokenizer):
    """Format example into chat template."""
    messages = example.get("messages", [])
    
    # Apply Qwen3 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def prepare_dataset(dataset, tokenizer):
    """Prepare dataset for training."""
    print("\nFormatting dataset with chat template...")
    
    formatted = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset["train"].column_names
    )
    
    # Show sample
    print("\n--- Sample formatted example ---")
    sample = formatted["train"][0]["text"]
    print(sample[:500] + "..." if len(sample) > 500 else sample)
    print("---\n")
    
    return formatted


# =============================================================================
# TRAINING
# =============================================================================

def train(model, tokenizer, dataset):
    """Train the model."""
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
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
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # Unsloth packing for efficiency
        dataset_text_field="text",
    )
    
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
    )
    
    # Show GPU memory before training
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated(0) / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Memory: {gpu_used:.2f}GB / {gpu_total:.2f}GB")
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    return trainer


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(model, tokenizer):
    """Save the trained model."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save LoRA adapters
    lora_path = OUTPUT_DIR / "lora_adapters"
    print(f"Saving LoRA adapters to: {lora_path}")
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    
    # Optionally merge and save (requires more VRAM)
    print("\nTo merge adapters with base model, run:")
    print("  python merge_model.py")
    
    # Save in GGUF format for deployment
    print("\nTo export to GGUF for llama.cpp/Ollama:")
    print("  python export_gguf.py")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DOCUMENT INTELLIGENCE MOE TRAINING")
    print("Qwen3-30B-A3B with Unsloth")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model()
    
    # Load and prepare data
    dataset = load_data()
    formatted_dataset = prepare_dataset(dataset, tokenizer)
    
    # Train
    trainer = train(model, tokenizer, formatted_dataset)
    
    # Save
    save_model(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Test: python test_moe.py")
    print("2. Export: python export_gguf.py")


if __name__ == "__main__":
    main()
