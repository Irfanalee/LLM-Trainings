#!/usr/bin/env python3
"""
TrOCR Fine-tuning for Whiteboard Handwriting Recognition
Supports both full fine-tuning and LoRA

Note: For most cases, pretrained TrOCR works well out of the box!
Only fine-tune if:
- Your handwriting style is very different from training data
- You need to recognize specific symbols/characters
- Accuracy on your test set is below 85%
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import load_metric
import numpy as np


class HandwritingDataset(Dataset):
    """Dataset for TrOCR fine-tuning"""

    def __init__(
        self,
        data_dir: str,
        processor: TrOCRProcessor,
        max_target_length: int = 128
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_target_length = max_target_length

        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.samples = json.load(f)
        else:
            # Try CSV format
            csv_path = self.data_dir / 'metadata.csv'
            self.samples = []
            with open(csv_path, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        self.samples.append({
                            'file_name': parts[0],
                            'text': parts[1].replace('\\,', ',')
                        })

        self.images_dir = self.data_dir / 'images'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_path = self.images_dir / sample['file_name']
        image = Image.open(image_path).convert('RGB')

        # Process image
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.squeeze()

        # Process text
        labels = self.processor.tokenizer(
            sample['text'],
            padding='max_length',
            max_length=self.max_target_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze()

        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'labels': labels
        }


def compute_cer(pred_ids, label_ids, processor):
    """Compute Character Error Rate"""
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Replace -100 in labels with pad token for decoding
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = 0
    for pred, label in zip(pred_str, label_str):
        # Simple CER calculation
        if len(label) == 0:
            continue
        distance = sum(c1 != c2 for c1, c2 in zip(pred, label)) + abs(len(pred) - len(label))
        cer += distance / len(label)

    return cer / len(pred_str) if pred_str else 0


def train_trocr(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    model_name: str = "microsoft/trocr-large-handwritten",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    """
    Fine-tune TrOCR for handwriting recognition

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        output_dir: Output directory for checkpoints
        model_name: Base model name
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        learning_rate: Learning rate
        use_lora: Use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
    """
    print("="*60)
    print("TrOCR Fine-tuning for Handwriting Recognition")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load processor and model
    print(f"\nLoading model: {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configure model
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Apply LoRA if requested
    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            print(f"\nApplying LoRA (r={lora_r}, alpha={lora_alpha})")

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        except ImportError:
            print("Warning: peft not installed, using full fine-tuning")
            use_lora = False

    model.to(device)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = HandwritingDataset(train_dir, processor)
    val_dataset = HandwritingDataset(val_dir, processor)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=3,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Set to "tensorboard" if you want logging
    )

    # Custom compute metrics
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Replace -100 with pad token for decoding
        labels_ids = np.where(labels_ids != -100, labels_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute CER
        total_chars = 0
        total_errors = 0
        for pred, label in zip(pred_str, label_str):
            total_chars += len(label)
            # Simple edit distance approximation
            errors = sum(c1 != c2 for c1, c2 in zip(pred, label)) + abs(len(pred) - len(label))
            total_errors += errors

        cer = total_errors / total_chars if total_chars > 0 else 0

        return {"cer": cer}

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving model...")
    if use_lora:
        model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    else:
        model.save_pretrained(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "processor"))

    print(f"\nTraining complete! Model saved to {output_dir}")

    return model, processor


def evaluate_trocr(
    model_dir: str,
    test_dir: str,
    batch_size: int = 8
):
    """
    Evaluate a fine-tuned TrOCR model
    """
    print("\nEvaluating TrOCR model...")

    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(os.path.join(model_dir, "processor"))

    model_path = os.path.join(model_dir, "final_model")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "lora_adapter")

    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    test_dataset = HandwritingDataset(test_dir, processor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']

            generated_ids = model.generate(pixel_values)

            all_preds.extend(generated_ids.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Decode and compute metrics
    pred_str = processor.batch_decode(all_preds, skip_special_tokens=True)

    # Replace -100 with pad token
    all_labels = np.array(all_labels)
    all_labels = np.where(all_labels != -100, all_labels, processor.tokenizer.pad_token_id)
    label_str = processor.batch_decode(all_labels, skip_special_tokens=True)

    # Compute CER and WER
    total_chars = sum(len(l) for l in label_str)
    total_words = sum(len(l.split()) for l in label_str)

    char_errors = sum(
        sum(c1 != c2 for c1, c2 in zip(p, l)) + abs(len(p) - len(l))
        for p, l in zip(pred_str, label_str)
    )

    word_errors = sum(
        sum(w1 != w2 for w1, w2 in zip(p.split(), l.split())) + abs(len(p.split()) - len(l.split()))
        for p, l in zip(pred_str, label_str)
    )

    cer = char_errors / total_chars if total_chars > 0 else 0
    wer = word_errors / total_words if total_words > 0 else 0

    print(f"\nResults:")
    print(f"  Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print(f"  Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")

    # Show some examples
    print("\nSample predictions:")
    for i in range(min(5, len(pred_str))):
        print(f"  Label: {label_str[i]}")
        print(f"  Pred:  {pred_str[i]}")
        print()

    return cer, wer


def main():
    parser = argparse.ArgumentParser(description='Fine-tune TrOCR for handwriting')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train TrOCR')
    train_parser.add_argument('--train-dir', required=True, help='Training data directory')
    train_parser.add_argument('--val-dir', required=True, help='Validation data directory')
    train_parser.add_argument('--output-dir', required=True, help='Output directory')
    train_parser.add_argument('--model', default='microsoft/trocr-large-handwritten',
                              help='Base model name')
    train_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    train_parser.add_argument('--lora', action='store_true', help='Use LoRA')
    train_parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    train_parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate TrOCR')
    eval_parser.add_argument('--model-dir', required=True, help='Model directory')
    eval_parser.add_argument('--test-dir', required=True, help='Test data directory')
    eval_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    if args.command == 'train':
        train_trocr(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
    elif args.command == 'eval':
        evaluate_trocr(
            model_dir=args.model_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# Example usage:
"""
# 1. Prepare data (IAM dataset or synthetic):
python scripts/prepare_datasets.py iam --input-dir datasets/iam --output-dir datasets/iam_processed

# 2. Train with full fine-tuning:
python training/train_trocr.py train \\
    --train-dir datasets/iam_processed/train \\
    --val-dir datasets/iam_processed/val \\
    --output-dir runs/trocr_whiteboard \\
    --epochs 10 \\
    --batch-size 8

# 3. Train with LoRA (memory efficient):
python training/train_trocr.py train \\
    --train-dir datasets/iam_processed/train \\
    --val-dir datasets/iam_processed/val \\
    --output-dir runs/trocr_lora \\
    --epochs 10 \\
    --batch-size 16 \\
    --lora

# 4. Evaluate:
python training/train_trocr.py eval \\
    --model-dir runs/trocr_whiteboard \\
    --test-dir datasets/iam_processed/test

# Memory requirements (RTX A4000 16GB):
# - Full fine-tuning: batch_size 4-8
# - LoRA fine-tuning: batch_size 8-16
"""
