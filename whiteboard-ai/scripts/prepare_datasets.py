#!/usr/bin/env python3
"""
Dataset Preparation Scripts
Convert downloaded datasets to training format
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from PIL import Image
import xml.etree.ElementTree as ET


def prepare_iam_for_trocr(
    iam_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> None:
    """
    Convert IAM Handwriting Database to TrOCR format

    Expected IAM structure:
    iam_dir/
    ├── words/
    │   └── a01/
    │       └── a01-000u/
    │           └── a01-000u-00-00.png
    └── words.txt (transcriptions)

    Output format:
    output_dir/
    ├── train/
    │   ├── images/
    │   └── metadata.csv
    ├── val/
    └── test/
    """
    iam_path = Path(iam_dir)
    output_path = Path(output_dir)

    # Read transcriptions
    words_file = iam_path / 'words.txt'
    if not words_file.exists():
        # Try alternative location
        words_file = iam_path / 'ascii' / 'words.txt'

    if not words_file.exists():
        print(f"Error: Cannot find words.txt in {iam_dir}")
        return

    samples = []

    with open(words_file, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Skip comments
                continue

            parts = line.strip().split()
            if len(parts) < 9:
                continue

            word_id = parts[0]  # e.g., a01-000u-00-00
            segmentation_result = parts[1]  # 'ok' or 'err'
            text = parts[8]  # The transcribed word

            # Skip erroneously segmented words
            if segmentation_result != 'ok':
                continue

            # Construct image path: a01-000u-00-00 -> words/a01/a01-000u/a01-000u-00-00.png
            parts = word_id.split('-')
            writer_id = parts[0]  # a01
            form_id = f"{parts[0]}-{parts[1]}"  # a01-000u

            image_path = iam_path / 'words' / writer_id / form_id / f"{word_id}.png"

            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'text': text
                })

    print(f"Found {len(samples)} valid samples")

    # Shuffle and split
    random.shuffle(samples)

    train_end = int(len(samples) * split_ratio[0])
    val_end = train_end + int(len(samples) * split_ratio[1])

    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }

    # Create output directories and copy files
    for split_name, split_samples in splits.items():
        split_dir = output_path / split_name
        images_dir = split_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        metadata = []

        for i, sample in enumerate(split_samples):
            src_path = Path(sample['image_path'])
            dst_filename = f"iam_{i:06d}.png"
            dst_path = images_dir / dst_filename

            shutil.copy2(src_path, dst_path)

            metadata.append({
                'file_name': dst_filename,
                'text': sample['text']
            })

        # Save metadata
        with open(split_dir / 'metadata.csv', 'w') as f:
            f.write('file_name,text\n')
            for item in metadata:
                # Escape commas in text
                text = item['text'].replace(',', '\\,')
                f.write(f"{item['file_name']},{text}\n")

        # Also save as JSON for easier processing
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  {split_name}: {len(split_samples)} samples")

    print(f"\nIAM dataset prepared in {output_path}")


def prepare_doclaynet_for_yolo(
    doclaynet_dir: str,
    output_dir: str,
    max_images: int = None
) -> None:
    """
    Convert DocLayNet to YOLO format

    DocLayNet has these classes:
    - Caption, Footnote, Formula, List-item, Page-footer, Page-header,
      Picture, Section-header, Table, Text, Title

    We'll map to whiteboard-relevant classes
    """
    doclaynet_path = Path(doclaynet_dir)
    output_path = Path(output_dir)

    # Class mapping: DocLayNet -> Whiteboard
    class_mapping = {
        'Title': 'header',
        'Section-header': 'header',
        'Text': 'text_block',
        'List-item': 'bullet_list',
        'Table': 'table',
        'Picture': 'diagram',
        'Formula': 'diagram',
        'Caption': 'text_block',
    }

    # Our output classes
    output_classes = ['header', 'text_block', 'bullet_list', 'diagram', 'table']
    class_to_id = {cls: i for i, cls in enumerate(output_classes)}

    # Create directories
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find annotation files
    annotations_dir = doclaynet_path / 'COCO'
    if not annotations_dir.exists():
        annotations_dir = doclaynet_path

    # Try different annotation file locations
    possible_ann_files = [
        doclaynet_path / 'COCO' / 'train.json',
        doclaynet_path / 'train.json',
        doclaynet_path / 'annotations' / 'train.json',
    ]

    ann_file = None
    for f in possible_ann_files:
        if f.exists():
            ann_file = f
            break

    if ann_file is None:
        print(f"Error: Cannot find annotation file in {doclaynet_dir}")
        print("Expected structure: COCO/train.json or train.json")
        return

    print(f"Loading annotations from {ann_file}")

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Create category mapping
    doclaynet_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Process images
    image_count = 0

    for img_info in coco_data['images']:
        if max_images and image_count >= max_images:
            break

        img_id = img_info['id']
        if img_id not in image_annotations:
            continue

        # Find image file
        img_filename = img_info['file_name']
        possible_paths = [
            doclaynet_path / 'PNG' / img_filename,
            doclaynet_path / 'images' / img_filename,
            doclaynet_path / img_filename,
        ]

        src_img_path = None
        for p in possible_paths:
            if p.exists():
                src_img_path = p
                break

        if src_img_path is None:
            continue

        # Copy image
        dst_img_name = f"doclaynet_{image_count:06d}.jpg"
        dst_img_path = images_dir / dst_img_name

        try:
            img = Image.open(src_img_path).convert('RGB')
            img.save(dst_img_path, quality=90)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error processing {src_img_path}: {e}")
            continue

        # Convert annotations to YOLO format
        yolo_annotations = []

        for ann in image_annotations[img_id]:
            category_name = doclaynet_categories.get(ann['category_id'], '')

            if category_name not in class_mapping:
                continue

            output_class = class_mapping[category_name]
            class_id = class_to_id[output_class]

            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Clamp values
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if yolo_annotations:
            # Save label file
            label_path = labels_dir / f"doclaynet_{image_count:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            image_count += 1

        if image_count % 1000 == 0:
            print(f"  Processed {image_count} images")

    # Save classes file
    with open(output_path / 'classes.txt', 'w') as f:
        for cls in output_classes:
            f.write(f"{cls}\n")

    # Create YAML config
    yaml_content = f"""# DocLayNet converted for Whiteboard Detection
path: {output_path}
train: images
val: images

names:
  0: header
  1: text_block
  2: bullet_list
  3: diagram
  4: table
"""

    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\nDocLayNet converted: {image_count} images -> {output_path}")


def create_train_val_split(
    dataset_dir: str,
    val_ratio: float = 0.2
) -> None:
    """
    Split a YOLO dataset into train and val sets
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    random.shuffle(image_files)

    val_count = int(len(image_files) * val_ratio)
    train_files = image_files[val_count:]
    val_files = image_files[:val_count]

    # Create train/val directories
    for split, files in [('train', train_files), ('val', val_files)]:
        split_images = dataset_path / split / 'images'
        split_labels = dataset_path / split / 'labels'
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Move image
            shutil.move(str(img_path), str(split_images / img_path.name))

            # Move label
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name
            if label_path.exists():
                shutil.move(str(label_path), str(split_labels / label_name))

        print(f"  {split}: {len(files)} images")

    # Clean up original directories if empty
    if not list(images_dir.glob('*')):
        images_dir.rmdir()
    if not list(labels_dir.glob('*')):
        labels_dir.rmdir()

    # Update YAML config
    yaml_path = dataset_path / 'dataset.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            content = f.read()

        content = content.replace('train: images', 'train: train/images')
        content = content.replace('val: images', 'val: val/images')

        with open(yaml_path, 'w') as f:
            f.write(content)

    print(f"\nDataset split complete!")


def prepare_action_items_for_lora(
    input_jsonl: str,
    output_dir: str,
    val_ratio: float = 0.1
) -> None:
    """
    Prepare action item dataset for Qwen2.5 LoRA training

    Input: JSONL with instruction/input/output format
    Output: Train/val splits in Alpaca format
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load samples
    samples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Shuffle and split
    random.shuffle(samples)
    val_count = int(len(samples) * val_ratio)

    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    # Save as JSON (Alpaca format for most trainers)
    with open(output_path / 'train.json', 'w') as f:
        json.dump(train_samples, f, indent=2)

    with open(output_path / 'val.json', 'w') as f:
        json.dump(val_samples, f, indent=2)

    # Also save as JSONL (for some trainers)
    with open(output_path / 'train.jsonl', 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    with open(output_path / 'val.jsonl', 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Action items dataset prepared:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # IAM preparation
    iam_parser = subparsers.add_parser('iam', help='Prepare IAM dataset for TrOCR')
    iam_parser.add_argument('--input-dir', required=True, help='Path to IAM dataset')
    iam_parser.add_argument('--output-dir', required=True, help='Output directory')

    # DocLayNet preparation
    doc_parser = subparsers.add_parser('doclaynet', help='Prepare DocLayNet for YOLO')
    doc_parser.add_argument('--input-dir', required=True, help='Path to DocLayNet dataset')
    doc_parser.add_argument('--output-dir', required=True, help='Output directory')
    doc_parser.add_argument('--max-images', type=int, help='Maximum images to process')

    # Split dataset
    split_parser = subparsers.add_parser('split', help='Split YOLO dataset into train/val')
    split_parser.add_argument('--dataset-dir', required=True, help='Dataset directory')
    split_parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio')

    # Action items
    action_parser = subparsers.add_parser('action', help='Prepare action items for LoRA')
    action_parser.add_argument('--input-jsonl', required=True, help='Input JSONL file')
    action_parser.add_argument('--output-dir', required=True, help='Output directory')

    args = parser.parse_args()

    if args.command == 'iam':
        prepare_iam_for_trocr(args.input_dir, args.output_dir)
    elif args.command == 'doclaynet':
        prepare_doclaynet_for_yolo(args.input_dir, args.output_dir, args.max_images)
    elif args.command == 'split':
        create_train_val_split(args.dataset_dir, args.val_ratio)
    elif args.command == 'action':
        prepare_action_items_for_lora(args.input_jsonl, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
