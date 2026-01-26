#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning for Whiteboard Region Detection
Optimized for RTX A4000 16GB
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def train_yolo_whiteboard(
    data_yaml: str,
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    output_dir: str = "./runs/whiteboard_detection",
    resume: bool = False,
    pretrained: str = None
):
    """
    Fine-tune YOLOv8 for whiteboard region detection

    Args:
        data_yaml: Path to dataset configuration YAML
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        imgsz: Image size for training
        output_dir: Output directory for checkpoints
        resume: Resume from last checkpoint
        pretrained: Path to pretrained model (default: yolov8{size}.pt)
    """

    print("="*60)
    print("YOLOv8 Whiteboard Region Detection Training")
    print("="*60)

    # Model selection
    if pretrained:
        model_path = pretrained
    else:
        model_path = f"yolov8{model_size}.pt"

    print(f"\nModel: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {imgsz}")

    # Load model
    model = YOLO(model_path)

    # Training configuration
    # Optimized for RTX A4000 with 16GB VRAM
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "project": output_dir,
        "name": f"yolo_whiteboard_{model_size}",

        # Optimization
        "optimizer": "AdamW",
        "lr0": 0.001,           # Initial learning rate
        "lrf": 0.01,            # Final learning rate (lr0 * lrf)
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # Warmup
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # Augmentation (good for whiteboards)
        "hsv_h": 0.015,         # HSV-Hue augmentation
        "hsv_s": 0.4,           # HSV-Saturation augmentation
        "hsv_v": 0.4,           # HSV-Value augmentation
        "degrees": 5.0,         # Rotation (+/- deg)
        "translate": 0.1,       # Translation (+/- fraction)
        "scale": 0.3,           # Scale (+/- gain)
        "shear": 2.0,           # Shear (+/- deg)
        "perspective": 0.0005,  # Perspective (+/- fraction)
        "flipud": 0.0,          # Flip up-down (not for text!)
        "fliplr": 0.5,          # Flip left-right
        "mosaic": 0.5,          # Mosaic augmentation
        "mixup": 0.1,           # Mixup augmentation

        # Loss weights
        "box": 7.5,             # Box loss weight
        "cls": 0.5,             # Class loss weight
        "dfl": 1.5,             # Distribution focal loss weight

        # Other
        "patience": 20,         # Early stopping patience
        "save": True,
        "save_period": 10,      # Save checkpoint every N epochs
        "cache": False,         # Set to True if RAM allows
        "device": 0,            # GPU device
        "workers": 8,           # DataLoader workers
        "amp": True,            # Automatic mixed precision
        "verbose": True,
        "resume": resume
    }

    # Train
    print("\nStarting training...")
    results = model.train(**train_args)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # Validate
    print("\nRunning validation...")
    val_results = model.val()

    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {output_dir}/yolo_whiteboard_{model_size}/weights/best.pt")

    return model, results


def export_model(model_path: str, formats: list = None):
    """
    Export trained model to various formats
    """
    if formats is None:
        formats = ["onnx", "torchscript"]

    model = YOLO(model_path)

    print("\nExporting model...")
    for fmt in formats:
        try:
            model.export(format=fmt)
            print(f"  Exported to {fmt}")
        except Exception as e:
            print(f"  Failed to export {fmt}: {e}")


def create_dataset_yaml(
    train_dir: str,
    val_dir: str,
    classes: list,
    output_path: str
):
    """
    Create a YOLO dataset configuration file
    """
    config = {
        "path": str(Path(train_dir).parent),
        "train": Path(train_dir).name,
        "val": Path(val_dir).name,
        "names": {i: name for i, name in enumerate(classes)}
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Dataset YAML saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for whiteboard detection')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (default: n)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16, reduce if OOM)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--output', type=str, default='./runs/whiteboard_detection',
                        help='Output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to custom pretrained model')
    parser.add_argument('--export', action='store_true',
                        help='Export model after training')

    args = parser.parse_args()

    # Train
    model, results = train_yolo_whiteboard(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        output_dir=args.output,
        resume=args.resume,
        pretrained=args.pretrained
    )

    # Export if requested
    if args.export:
        best_model = Path(args.output) / f"yolo_whiteboard_{args.model}" / "weights" / "best.pt"
        if best_model.exists():
            export_model(str(best_model))


if __name__ == "__main__":
    main()


# Example usage:
"""
# 1. First generate synthetic data:
python scripts/generate_synthetic_data.py --output-dir datasets/synthetic --num-whiteboards 1000

# 2. Train YOLOv8:
python training/train_yolo.py \\
    --data datasets/synthetic/whiteboard_yolo.yaml \\
    --model n \\
    --epochs 100 \\
    --batch 16

# 3. For RTX A4000, you can try larger models:
python training/train_yolo.py \\
    --data datasets/synthetic/whiteboard_yolo.yaml \\
    --model s \\
    --epochs 100 \\
    --batch 8

# Memory requirements:
# - yolov8n: ~4GB VRAM, batch 16-32
# - yolov8s: ~6GB VRAM, batch 8-16
# - yolov8m: ~8GB VRAM, batch 4-8
# - yolov8l: ~12GB VRAM, batch 2-4
# - yolov8x: ~16GB VRAM, batch 1-2
"""
