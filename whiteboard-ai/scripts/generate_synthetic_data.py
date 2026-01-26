#!/usr/bin/env python3
"""
Synthetic Data Generation for Whiteboard AI
Generates:
1. Fake whiteboard images with annotations (for YOLOv8)
2. Handwriting image-text pairs (for TrOCR fine-tuning)
3. Meeting notes with action items (for Qwen2.5 LoRA)
"""

import random
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import argparse

# Configuration
WHITEBOARD_COLORS = [
    (255, 255, 255),  # Pure white
    (250, 250, 245),  # Off-white
    (245, 245, 240),  # Cream
    (240, 248, 255),  # Alice blue
]

MARKER_COLORS = [
    (0, 0, 0),        # Black
    (0, 0, 139),      # Dark blue
    (139, 0, 0),      # Dark red
    (0, 100, 0),      # Dark green
]

# Sample meeting content
MEETING_TOPICS = [
    "Q4 Budget Review", "Sprint Planning", "Product Roadmap",
    "Customer Feedback", "Tech Debt", "Team Building",
    "Marketing Strategy", "Hiring Plan", "Release Planning"
]

TASK_TEMPLATES = [
    "Review {item}",
    "Update {item}",
    "Schedule meeting with {person}",
    "Send {item} to {person}",
    "Complete {item} analysis",
    "Prepare {item} report",
    "Fix {item} issue",
    "Research {item} options",
    "Create {item} documentation",
    "Follow up with {person} about {item}"
]

PEOPLE = [
    "Sarah", "John", "Mike", "Lisa", "David", "Emma",
    "Alex", "Chris", "Taylor", "Jordan", "Team Lead", "PM"
]

ITEMS = [
    "Q4 budget", "design mockups", "API docs", "test results",
    "customer feedback", "sprint backlog", "deployment plan",
    "security audit", "performance metrics", "user research"
]

DEADLINES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "next week", "EOD", "by noon", "ASAP", "Jan 15", "Jan 20",
    "end of month", "Q1", "before standup"
]

PRIORITIES = ["High", "Normal", "Low"]


@dataclass
class SyntheticRegion:
    """A region on the whiteboard"""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    text: str


@dataclass
class ActionItem:
    """An action item from meeting notes"""
    task: str
    assignee: str
    deadline: str
    priority: str


def get_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Get a font, trying multiple options"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue

    return ImageFont.load_default()


def generate_handwriting_style(text: str, draw: ImageDraw.Draw,
                                x: int, y: int, font: ImageFont.FreeTypeFont,
                                color: Tuple[int, int, int]) -> Tuple[int, int]:
    """Draw text with slight variations to simulate handwriting"""
    char_x = x
    max_y = y

    for char in text:
        # Add slight randomness to position
        offset_x = random.randint(-1, 1)
        offset_y = random.randint(-2, 2)

        draw.text((char_x + offset_x, y + offset_y), char, fill=color, font=font)

        # Get character width
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_x += char_width + random.randint(-1, 2)
        max_y = max(max_y, y + offset_y + bbox[3])

    return char_x, max_y


def generate_whiteboard_image(
    output_path: str,
    width: int = 1200,
    height: int = 900,
    num_regions: int = None
) -> List[SyntheticRegion]:
    """
    Generate a synthetic whiteboard image with regions
    Returns list of regions for YOLO annotation
    """
    if num_regions is None:
        num_regions = random.randint(3, 6)

    # Create base whiteboard
    bg_color = random.choice(WHITEBOARD_COLORS)
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    # Add slight noise/texture
    pixels = np.array(image)
    noise = np.random.normal(0, 3, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)

    regions = []
    margin = 50

    # Generate regions with a grid-based layout
    region_types = ['header', 'text_block', 'bullet_list', 'diagram', 'action_item']

    # Always add a header
    header_font = get_font(36)
    topic = random.choice(MEETING_TOPICS)
    header_bbox = draw.textbbox((0, 0), topic, font=header_font)
    header_width = header_bbox[2] - header_bbox[0]
    header_x = (width - header_width) // 2

    draw.text((header_x, margin), topic, fill=random.choice(MARKER_COLORS), font=header_font)
    regions.append(SyntheticRegion(
        x1=header_x - 10,
        y1=margin - 5,
        x2=header_x + header_width + 10,
        y2=margin + header_bbox[3] + 5,
        label="header",
        text=topic
    ))

    # Add content regions
    content_y = margin + header_bbox[3] + 40
    content_font = get_font(20)
    small_font = get_font(16)

    # Split remaining space into columns
    col_width = (width - 3 * margin) // 2

    for region_idx in range(num_regions - 1):
        region_type = random.choice(region_types[1:])  # Skip header

        col = region_idx % 2
        region_x = margin + col * (col_width + margin)
        region_y = content_y + (region_idx // 2) * 200

        if region_y + 150 > height - margin:
            continue

        color = random.choice(MARKER_COLORS)

        if region_type == 'text_block':
            # Generate random text block
            lines = []
            for _ in range(random.randint(2, 4)):
                line = random.choice([
                    f"Discussed {random.choice(ITEMS)}",
                    f"{random.choice(PEOPLE)} presented updates",
                    f"Need to review {random.choice(ITEMS)}",
                    f"Concerns about {random.choice(ITEMS)}"
                ])
                lines.append(line)

            text = "\n".join(lines)
            y_offset = 0
            max_width = 0

            for line in lines:
                draw.text((region_x, region_y + y_offset), line, fill=color, font=content_font)
                bbox = draw.textbbox((0, 0), line, font=content_font)
                max_width = max(max_width, bbox[2] - bbox[0])
                y_offset += bbox[3] - bbox[1] + 5

            regions.append(SyntheticRegion(
                x1=region_x - 5,
                y1=region_y - 5,
                x2=region_x + max_width + 10,
                y2=region_y + y_offset + 5,
                label="text_block",
                text=text
            ))

        elif region_type == 'bullet_list':
            # Generate bullet list
            items = []
            y_offset = 0
            max_width = 0

            for i in range(random.randint(3, 5)):
                item = random.choice([
                    random.choice(ITEMS).capitalize(),
                    f"Review {random.choice(ITEMS)}",
                    f"{random.choice(PEOPLE)}'s input"
                ])
                items.append(item)
                bullet_text = f"• {item}"
                draw.text((region_x, region_y + y_offset), bullet_text, fill=color, font=content_font)
                bbox = draw.textbbox((0, 0), bullet_text, font=content_font)
                max_width = max(max_width, bbox[2] - bbox[0])
                y_offset += bbox[3] - bbox[1] + 8

            regions.append(SyntheticRegion(
                x1=region_x - 5,
                y1=region_y - 5,
                x2=region_x + max_width + 10,
                y2=region_y + y_offset + 5,
                label="bullet_list",
                text="\n".join(items)
            ))

        elif region_type == 'action_item':
            # Generate action items
            y_offset = 0
            max_width = 0
            text_lines = []

            # Draw "ACTION ITEMS" header
            draw.text((region_x, region_y), "ACTION ITEMS:", fill=color, font=content_font)
            bbox = draw.textbbox((0, 0), "ACTION ITEMS:", font=content_font)
            y_offset += bbox[3] - bbox[1] + 10

            for _ in range(random.randint(2, 4)):
                task = random.choice(TASK_TEMPLATES).format(
                    item=random.choice(ITEMS),
                    person=random.choice(PEOPLE)
                )
                assignee = random.choice(PEOPLE)
                deadline = random.choice(DEADLINES)

                line = f"[ ] {task} - @{assignee} by {deadline}"
                text_lines.append(line)
                draw.text((region_x, region_y + y_offset), line, fill=color, font=small_font)
                bbox = draw.textbbox((0, 0), line, font=small_font)
                max_width = max(max_width, bbox[2] - bbox[0])
                y_offset += bbox[3] - bbox[1] + 8

            regions.append(SyntheticRegion(
                x1=region_x - 5,
                y1=region_y - 5,
                x2=region_x + max_width + 10,
                y2=region_y + y_offset + 5,
                label="action_item",
                text="\n".join(text_lines)
            ))

        elif region_type == 'diagram':
            # Simple box diagram
            box_width = random.randint(80, 120)
            box_height = random.randint(40, 60)

            # Draw 2-3 connected boxes
            boxes = []
            for i in range(random.randint(2, 3)):
                bx = region_x + i * (box_width + 30)
                by = region_y + random.randint(0, 30)
                boxes.append((bx, by))
                draw.rectangle([bx, by, bx + box_width, by + box_height], outline=color, width=2)
                label = random.choice(["Start", "Process", "End", "Review", "Deploy"])
                draw.text((bx + 10, by + 15), label, fill=color, font=small_font)

                # Draw arrow to next box
                if i < 2:
                    arrow_start = (bx + box_width, by + box_height // 2)
                    arrow_end = (bx + box_width + 30, by + box_height // 2)
                    draw.line([arrow_start, arrow_end], fill=color, width=2)

            if boxes:
                min_x = min(b[0] for b in boxes) - 5
                min_y = min(b[1] for b in boxes) - 5
                max_x = max(b[0] for b in boxes) + box_width + 5
                max_y = max(b[1] for b in boxes) + box_height + 5

                regions.append(SyntheticRegion(
                    x1=min_x,
                    y1=min_y,
                    x2=max_x,
                    y2=max_y,
                    label="diagram",
                    text="diagram"
                ))

    # Save image
    image.save(output_path, quality=95)
    return regions


def convert_to_yolo_format(
    regions: List[SyntheticRegion],
    image_width: int,
    image_height: int,
    class_mapping: Dict[str, int]
) -> List[str]:
    """
    Convert regions to YOLO annotation format
    Format: class_id x_center y_center width height (all normalized 0-1)
    """
    annotations = []

    for region in regions:
        if region.label not in class_mapping:
            continue

        class_id = class_mapping[region.label]

        # Calculate normalized coordinates
        x_center = ((region.x1 + region.x2) / 2) / image_width
        y_center = ((region.y1 + region.y2) / 2) / image_height
        width = (region.x2 - region.x1) / image_width
        height = (region.y2 - region.y1) / image_height

        # Clamp values
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return annotations


def generate_action_item_dataset(
    output_path: str,
    num_samples: int = 5000
) -> None:
    """
    Generate meeting notes with action items for LLM fine-tuning
    Output: JSONL file with instruction-response pairs
    """
    samples = []

    for _ in range(num_samples):
        # Generate meeting notes
        topic = random.choice(MEETING_TOPICS)
        num_discussion_points = random.randint(2, 5)
        num_action_items = random.randint(2, 5)

        # Create discussion points
        discussion = []
        for i in range(num_discussion_points):
            point = random.choice([
                f"Reviewed {random.choice(ITEMS)} with the team",
                f"{random.choice(PEOPLE)} shared updates on {random.choice(ITEMS)}",
                f"Discussion about {random.choice(ITEMS)} timeline",
                f"Need input from {random.choice(PEOPLE)} on {random.choice(ITEMS)}",
                f"Concerns raised about {random.choice(ITEMS)}",
                f"Agreed to prioritize {random.choice(ITEMS)}"
            ])
            discussion.append(f"{i+1}. {point}")

        # Create action items
        action_items = []
        action_text = []
        for _ in range(num_action_items):
            task = random.choice(TASK_TEMPLATES).format(
                item=random.choice(ITEMS),
                person=random.choice(PEOPLE)
            )
            assignee = random.choice(PEOPLE)
            deadline = random.choice(DEADLINES)
            priority = random.choice(PRIORITIES)

            action_items.append({
                "task": task,
                "assignee": assignee,
                "deadline": deadline,
                "priority": priority
            })

            # Add to text in various formats
            format_choice = random.randint(0, 3)
            if format_choice == 0:
                action_text.append(f"- {task} (@{assignee}, due {deadline})")
            elif format_choice == 1:
                action_text.append(f"TODO: {task} - {assignee} by {deadline}")
            elif format_choice == 2:
                action_text.append(f"Action: {task}. Owner: {assignee}. Deadline: {deadline}")
            else:
                action_text.append(f"[ ] {task} ({assignee}) - {deadline}")

        # Combine into meeting notes
        meeting_notes = f"""Meeting: {topic}

Discussion:
{chr(10).join(discussion)}

Action Items:
{chr(10).join(action_text)}
"""

        # Create training sample
        sample = {
            "instruction": "Extract all action items from the following meeting notes. Return a JSON array with task, assignee, deadline, and priority for each item.",
            "input": meeting_notes,
            "output": json.dumps(action_items, indent=2)
        }

        samples.append(sample)

    # Save as JSONL
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Generated {num_samples} action item samples -> {output_path}")


def generate_handwriting_pairs(
    output_dir: str,
    num_samples: int = 1000
) -> None:
    """
    Generate handwriting images with ground truth text
    For TrOCR fine-tuning
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # Sample text lines
    text_samples = [
        "Review budget proposal",
        "Meeting with Sarah at 3pm",
        "Update documentation",
        "Call client about project",
        "Sprint planning tomorrow",
        "Fix authentication bug",
        "Prepare presentation slides",
        "Send report to team lead",
        "Schedule code review",
        "Deploy to staging server"
    ] + [
        f"{random.choice(['Review', 'Update', 'Fix', 'Check', 'Send'])} {random.choice(ITEMS)}"
        for _ in range(50)
    ] + [
        f"{random.choice(PEOPLE)} - {random.choice(DEADLINES)}"
        for _ in range(50)
    ]

    annotations = []

    for i in range(num_samples):
        # Select text
        text = random.choice(text_samples)

        # Create image
        font_size = random.randint(24, 36)
        font = get_font(font_size)

        # Calculate image size
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Add padding
        padding = 20
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        # Create image with whiteboard background
        bg_color = random.choice(WHITEBOARD_COLORS)
        image = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(image)

        # Add slight noise
        pixels = np.array(image)
        noise = np.random.normal(0, 5, pixels.shape).astype(np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)

        # Draw text
        color = random.choice(MARKER_COLORS)
        draw.text((padding, padding), text, fill=color, font=font)

        # Save image
        image_filename = f"handwriting_{i:05d}.png"
        image_path = os.path.join(output_dir, 'images', image_filename)
        image.save(image_path)

        annotations.append({
            "image": image_filename,
            "text": text
        })

    # Save annotations
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    # Also save as CSV for TrOCR
    with open(os.path.join(output_dir, 'labels.csv'), 'w') as f:
        f.write("image,text\n")
        for ann in annotations:
            f.write(f"{ann['image']},{ann['text']}\n")

    print(f"Generated {num_samples} handwriting samples -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output-dir', type=str, default='./datasets/synthetic',
                        help='Output directory for generated data')
    parser.add_argument('--num-whiteboards', type=int, default=500,
                        help='Number of whiteboard images to generate')
    parser.add_argument('--num-action-items', type=int, default=5000,
                        help='Number of action item samples to generate')
    parser.add_argument('--num-handwriting', type=int, default=1000,
                        help='Number of handwriting samples to generate')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class mapping for YOLO
    class_mapping = {
        'header': 0,
        'text_block': 1,
        'bullet_list': 2,
        'action_item': 3,
        'diagram': 4
    }

    # Save class mapping
    with open(output_dir / 'classes.txt', 'w') as f:
        for cls_name in class_mapping.keys():
            f.write(f"{cls_name}\n")

    # Generate whiteboard images for YOLO
    print("\n" + "="*60)
    print("Generating Whiteboard Images for YOLOv8...")
    print("="*60)

    images_dir = output_dir / 'whiteboards' / 'images'
    labels_dir = output_dir / 'whiteboards' / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_whiteboards):
        image_path = images_dir / f"whiteboard_{i:04d}.jpg"
        regions = generate_whiteboard_image(str(image_path))

        # Convert to YOLO format
        annotations = convert_to_yolo_format(regions, 1200, 900, class_mapping)

        # Save annotations
        label_path = labels_dir / f"whiteboard_{i:04d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.num_whiteboards} whiteboards")

    print(f"Whiteboard images saved to {images_dir}")

    # Generate action item dataset
    print("\n" + "="*60)
    print("Generating Action Item Dataset for Qwen2.5...")
    print("="*60)

    action_items_path = output_dir / 'action_items.jsonl'
    generate_action_item_dataset(str(action_items_path), args.num_action_items)

    # Generate handwriting dataset
    print("\n" + "="*60)
    print("Generating Handwriting Dataset for TrOCR...")
    print("="*60)

    handwriting_dir = output_dir / 'handwriting'
    generate_handwriting_pairs(str(handwriting_dir), args.num_handwriting)

    # Create YAML config for YOLO
    yolo_config = f"""# Whiteboard Dataset Configuration
path: {output_dir / 'whiteboards'}
train: images
val: images

names:
  0: header
  1: text_block
  2: bullet_list
  3: action_item
  4: diagram
"""

    with open(output_dir / 'whiteboard_yolo.yaml', 'w') as f:
        f.write(yolo_config)

    print(f"\nYOLO config saved to {output_dir / 'whiteboard_yolo.yaml'}")

    # Summary
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated:")
    print(f"  - {args.num_whiteboards} whiteboard images with YOLO annotations")
    print(f"  - {args.num_action_items} action item training samples")
    print(f"  - {args.num_handwriting} handwriting image-text pairs")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
