#!/usr/bin/env python3
"""
Evaluation Scripts for Whiteboard AI Pipeline
Measures performance of each component
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from PIL import Image
import numpy as np
from collections import defaultdict


def evaluate_region_detection(
    model_path: str,
    test_images_dir: str,
    test_labels_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate YOLOv8 region detection

    Metrics:
    - mAP@0.5 (mean Average Precision)
    - Precision/Recall per class
    - Inference time
    """
    from ultralytics import YOLO
    import time

    print("\n" + "="*60)
    print("Evaluating Region Detection (YOLOv8)")
    print("="*60)

    model = YOLO(model_path)

    # Get class names
    class_names = model.names

    # Prepare results storage
    results = {
        'total_images': 0,
        'total_predictions': 0,
        'total_ground_truth': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_class': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
        'inference_times': []
    }

    images_dir = Path(test_images_dir)
    labels_dir = Path(test_labels_dir)

    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    for image_path in image_files:
        results['total_images'] += 1

        # Load ground truth
        label_path = labels_dir / (image_path.stem + '.txt')
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append({
                            'class': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })

        results['total_ground_truth'] += len(gt_boxes)

        # Run inference
        start_time = time.time()
        predictions = model(str(image_path), conf=conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        results['inference_times'].append(inference_time)

        # Get image dimensions
        img = Image.open(image_path)
        img_w, img_h = img.size

        # Process predictions
        pred_boxes = []
        for result in predictions:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Convert to normalized center format
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                pred_boxes.append({
                    'class': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })

        results['total_predictions'] += len(pred_boxes)

        # Match predictions to ground truth
        matched_gt = set()
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if pred['class'] != gt['class']:
                    continue

                # Calculate IoU
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                results['true_positives'] += 1
                results['per_class'][pred['class']]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                results['false_positives'] += 1
                results['per_class'][pred['class']]['fp'] += 1

        # Count false negatives
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                results['false_negatives'] += 1
                results['per_class'][gt['class']]['fn'] += 1

    # Calculate metrics
    precision = results['true_positives'] / (results['true_positives'] + results['false_positives']) if (results['true_positives'] + results['false_positives']) > 0 else 0
    recall = results['true_positives'] / (results['true_positives'] + results['false_negatives']) if (results['true_positives'] + results['false_negatives']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_inference_time = np.mean(results['inference_times']) if results['inference_times'] else 0

    print(f"\nResults:")
    print(f"  Images evaluated: {results['total_images']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Avg Inference Time: {avg_inference_time*1000:.2f} ms")

    print("\nPer-class results:")
    for class_id, metrics in results['per_class'].items():
        class_name = class_names.get(class_id, f"class_{class_id}")
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {class_name}: P={p:.3f}, R={r:.3f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time
    }


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate IoU between two boxes in center format"""
    # Convert to corners
    b1_x1 = box1['x_center'] - box1['width'] / 2
    b1_y1 = box1['y_center'] - box1['height'] / 2
    b1_x2 = box1['x_center'] + box1['width'] / 2
    b1_y2 = box1['y_center'] + box1['height'] / 2

    b2_x1 = box2['x_center'] - box2['width'] / 2
    b2_y1 = box2['y_center'] - box2['height'] / 2
    b2_x2 = box2['x_center'] + box2['width'] / 2
    b2_y2 = box2['y_center'] + box2['height'] / 2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Union
    b1_area = box1['width'] * box1['height']
    b2_area = box2['width'] * box2['height']
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def evaluate_ocr(
    model_path: str,
    test_data_dir: str,
    processor_path: str = None
) -> Dict:
    """
    Evaluate TrOCR OCR performance

    Metrics:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Exact match accuracy
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import time

    print("\n" + "="*60)
    print("Evaluating OCR (TrOCR)")
    print("="*60)

    # Load model
    if processor_path:
        processor = TrOCRProcessor.from_pretrained(processor_path)
    else:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')

    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    test_dir = Path(test_data_dir)
    metadata_path = test_dir / 'metadata.json'

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            samples = json.load(f)
    else:
        # Try CSV
        csv_path = test_dir / 'metadata.csv'
        samples = []
        with open(csv_path, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    samples.append({'file_name': parts[0], 'text': parts[1]})

    # Evaluate
    results = {
        'total_samples': len(samples),
        'total_chars': 0,
        'total_words': 0,
        'char_errors': 0,
        'word_errors': 0,
        'exact_matches': 0,
        'inference_times': []
    }

    images_dir = test_dir / 'images'

    for sample in samples:
        image_path = images_dir / sample['file_name']
        if not image_path.exists():
            continue

        ground_truth = sample['text']

        # Run OCR
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        inference_time = time.time() - start_time
        results['inference_times'].append(inference_time)

        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate metrics
        results['total_chars'] += len(ground_truth)
        results['total_words'] += len(ground_truth.split())

        # Character errors (simple edit distance approximation)
        char_errors = sum(c1 != c2 for c1, c2 in zip(predicted, ground_truth)) + abs(len(predicted) - len(ground_truth))
        results['char_errors'] += char_errors

        # Word errors
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        word_errors = sum(w1 != w2 for w1, w2 in zip(pred_words, gt_words)) + abs(len(pred_words) - len(gt_words))
        results['word_errors'] += word_errors

        # Exact match
        if predicted.strip().lower() == ground_truth.strip().lower():
            results['exact_matches'] += 1

    # Calculate final metrics
    cer = results['char_errors'] / results['total_chars'] if results['total_chars'] > 0 else 0
    wer = results['word_errors'] / results['total_words'] if results['total_words'] > 0 else 0
    exact_match_rate = results['exact_matches'] / results['total_samples'] if results['total_samples'] > 0 else 0
    avg_inference_time = np.mean(results['inference_times']) if results['inference_times'] else 0

    print(f"\nResults:")
    print(f"  Samples evaluated: {results['total_samples']}")
    print(f"  Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print(f"  Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"  Exact Match Rate: {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")
    print(f"  Avg Inference Time: {avg_inference_time*1000:.2f} ms")

    return {
        'cer': cer,
        'wer': wer,
        'exact_match_rate': exact_match_rate,
        'avg_inference_time': avg_inference_time
    }


def evaluate_action_extraction(
    model_path: str,
    test_data: str,
    tokenizer_path: str = None,
    use_lora: bool = True
) -> Dict:
    """
    Evaluate action item extraction

    Metrics:
    - Precision/Recall/F1 for action items
    - JSON parse success rate
    - Field accuracy (task, assignee, deadline)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import time

    print("\n" + "="*60)
    print("Evaluating Action Item Extraction (Qwen2.5)")
    print("="*60)

    # Load model
    base_model = "Qwen/Qwen2.5-7B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    if use_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)

    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.eval()

    # Load test data
    if test_data.endswith('.jsonl'):
        with open(test_data, 'r') as f:
            samples = [json.loads(line) for line in f]
    else:
        with open(test_data, 'r') as f:
            samples = json.load(f)

    # Limit samples for evaluation
    samples = samples[:100]  # Evaluate on 100 samples max

    results = {
        'total_samples': len(samples),
        'parse_success': 0,
        'total_gt_items': 0,
        'total_pred_items': 0,
        'matched_items': 0,
        'task_matches': 0,
        'assignee_matches': 0,
        'deadline_matches': 0,
        'inference_times': []
    }

    for sample in samples:
        gt_output = sample['output']

        # Parse ground truth
        try:
            gt_items = json.loads(gt_output)
        except:
            continue

        results['total_gt_items'] += len(gt_items)

        # Create prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts action items from meeting notes. Always return valid JSON."},
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.95
            )
        inference_time = time.time() - start_time
        results['inference_times'].append(inference_time)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse prediction
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                pred_items = json.loads(json_match.group())
                results['parse_success'] += 1
                results['total_pred_items'] += len(pred_items)

                # Match items
                for pred in pred_items:
                    for gt in gt_items:
                        if pred.get('task', '').lower() in gt.get('task', '').lower() or \
                           gt.get('task', '').lower() in pred.get('task', '').lower():
                            results['matched_items'] += 1
                            if pred.get('assignee', '').lower() == gt.get('assignee', '').lower():
                                results['assignee_matches'] += 1
                            if pred.get('deadline', '').lower() == gt.get('deadline', '').lower():
                                results['deadline_matches'] += 1
                            break
        except:
            pass

    # Calculate metrics
    parse_rate = results['parse_success'] / results['total_samples'] if results['total_samples'] > 0 else 0
    precision = results['matched_items'] / results['total_pred_items'] if results['total_pred_items'] > 0 else 0
    recall = results['matched_items'] / results['total_gt_items'] if results['total_gt_items'] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    assignee_acc = results['assignee_matches'] / results['matched_items'] if results['matched_items'] > 0 else 0
    deadline_acc = results['deadline_matches'] / results['matched_items'] if results['matched_items'] > 0 else 0
    avg_inference_time = np.mean(results['inference_times']) if results['inference_times'] else 0

    print(f"\nResults:")
    print(f"  Samples evaluated: {results['total_samples']}")
    print(f"  JSON Parse Rate: {parse_rate:.4f} ({parse_rate*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Assignee Accuracy: {assignee_acc:.4f} ({assignee_acc*100:.2f}%)")
    print(f"  Deadline Accuracy: {deadline_acc:.4f} ({deadline_acc*100:.2f}%)")
    print(f"  Avg Inference Time: {avg_inference_time*1000:.2f} ms")

    return {
        'parse_rate': parse_rate,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'assignee_accuracy': assignee_acc,
        'deadline_accuracy': deadline_acc,
        'avg_inference_time': avg_inference_time
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Whiteboard AI components')
    subparsers = parser.add_subparsers(dest='command')

    # Region detection
    yolo_parser = subparsers.add_parser('yolo', help='Evaluate YOLO region detection')
    yolo_parser.add_argument('--model', required=True, help='Model path')
    yolo_parser.add_argument('--images', required=True, help='Test images directory')
    yolo_parser.add_argument('--labels', required=True, help='Test labels directory')

    # OCR
    ocr_parser = subparsers.add_parser('ocr', help='Evaluate TrOCR')
    ocr_parser.add_argument('--model', required=True, help='Model path')
    ocr_parser.add_argument('--data', required=True, help='Test data directory')
    ocr_parser.add_argument('--processor', help='Processor path')

    # Action extraction
    action_parser = subparsers.add_parser('action', help='Evaluate action extraction')
    action_parser.add_argument('--model', required=True, help='Model/adapter path')
    action_parser.add_argument('--data', required=True, help='Test data file')
    action_parser.add_argument('--tokenizer', help='Tokenizer path')
    action_parser.add_argument('--no-lora', action='store_true', help='Model is not LoRA')

    args = parser.parse_args()

    if args.command == 'yolo':
        evaluate_region_detection(args.model, args.images, args.labels)
    elif args.command == 'ocr':
        evaluate_ocr(args.model, args.data, args.processor)
    elif args.command == 'action':
        evaluate_action_extraction(args.model, args.data, args.tokenizer, not args.no_lora)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
