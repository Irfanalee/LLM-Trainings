"""
Fine-Tuning Script for Document Intelligence Agent
Train/fine-tune the models on your own document datasets

This script supports:
1. LoRA fine-tuning of Qwen2.5 for domain-specific reasoning
2. Creating custom training data from your documents

Perfect for specializing on:
- Your company's document formats
- Industry-specific terminology
- Custom extraction schemas
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = True
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    
    # Output
    output_dir: str = "models/checkpoints"
    save_steps: int = 100
    logging_steps: int = 10
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Qwen
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


def prepare_training_data(
    data_dir: str,
    output_path: str = "data/train/training_data.jsonl"
) -> str:
    """
    Prepare training data from documents and annotations
    
    Expected input format in data_dir:
    - documents/: folder with document images/PDFs
    - annotations.json: file with Q&A pairs or extractions
    
    Annotation format:
    [
        {
            "document": "invoice_001.png",
            "conversations": [
                {"role": "user", "content": "What is the invoice number?"},
                {"role": "assistant", "content": "The invoice number is INV-2024-001."}
            ]
        },
        ...
    ]
    """
    print("Preparing training data...")
    
    data_dir = Path(data_dir)
    annotations_path = data_dir / "annotations.json"
    
    if not annotations_path.exists():
        print(f"Creating sample annotations file at: {annotations_path}")
        sample_annotations = [
            {
                "document": "sample_invoice.png",
                "conversations": [
                    {
                        "role": "user",
                        "content": "What is the invoice number and total amount?"
                    },
                    {
                        "role": "assistant", 
                        "content": "The invoice number is INV-2024-001 and the total amount is $1,250.00."
                    }
                ]
            },
            {
                "document": "sample_invoice.png",
                "conversations": [
                    {
                        "role": "user",
                        "content": "Extract the customer name and payment due date."
                    },
                    {
                        "role": "assistant",
                        "content": "Customer: Acme Corporation\nPayment Due: February 15, 2024"
                    }
                ]
            }
        ]
        
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(annotations_path, 'w') as f:
            json.dump(sample_annotations, f, indent=2)
        
        print(f"Sample annotations created. Edit this file with your own training data.")
        return None
    
    # Load and process annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Convert to training format
    training_examples = []
    
    for ann in annotations:
        # Build the training example
        conversations = ann.get("conversations", [])
        
        if len(conversations) >= 2:
            # Format as instruction-following
            user_msg = conversations[0]["content"]
            assistant_msg = conversations[1]["content"]
            
            # Add document context prompt
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful document analysis assistant. Analyze documents and provide accurate, structured responses."
                    },
                    {
                        "role": "user",
                        "content": f"Based on the document analysis, {user_msg}"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_msg
                    }
                ]
            }
            training_examples.append(example)
    
    # Save training data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Prepared {len(training_examples)} training examples")
    print(f"Saved to: {output_path}")
    
    return str(output_path)


def load_model_for_training(config: TrainingConfig):
    """Load model with LoRA adapters for training"""
    print("Loading model for training...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization config
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def train(
    config: TrainingConfig,
    train_data_path: str,
    val_data_path: Optional[str] = None
):
    """Run fine-tuning with the specified configuration"""
    print("=" * 50)
    print("Starting Fine-Tuning")
    print("=" * 50)
    
    from transformers import TrainingArguments, Trainer
    from datasets import load_dataset
    
    # Load model
    model, tokenizer = load_model_for_training(config)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files={"train": train_data_path})
    
    if val_data_path:
        val_dataset = load_dataset("json", data_files={"validation": val_data_path})
        dataset["validation"] = val_dataset["validation"]
    
    # Tokenize function
    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Set labels equal to input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit" if config.use_4bit else "adamw_torch",
        report_to="none",  # Disable wandb etc.
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        tokenizer=tokenizer,
    )
    
    # Train!
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_path = Path("models/final") / f"document_agent_{timestamp}"
    print(f"\nSaving model to: {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"\nModel saved to: {final_path}")
    print("\nTo use the fine-tuned model:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_path}')")
    
    return str(final_path)


def main():
    """Main entry point for fine-tuning"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Document Intelligence Agent")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Directory containing documents and annotations")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare training data, don't train")
    
    args = parser.parse_args()
    
    # Prepare training data
    train_data = prepare_training_data(args.data_dir)
    
    if args.prepare_only or train_data is None:
        print("\nTraining data preparation complete.")
        print("Edit data/raw/annotations.json with your training examples, then run again.")
        return
    
    # Configure training
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Run training
    train(config, train_data)


if __name__ == "__main__":
    main()
