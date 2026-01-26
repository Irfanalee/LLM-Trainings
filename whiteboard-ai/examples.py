"""
Whiteboard Meeting Notes AI - Usage Examples
Demonstrates complete functionality with sample code
"""

from whiteboard_ai import WhiteboardAI
import torch

# =============================================================================
# Example 1: Basic Whiteboard Analysis
# =============================================================================

print("="*80)
print("EXAMPLE 1: Basic Whiteboard Analysis")
print("="*80)

# Initialize the AI
ai = WhiteboardAI(device="cuda" if torch.cuda.is_available() else "cpu")

# Analyze a whiteboard image
# Replace with your actual image path
image_path = "sample_whiteboard.jpg"

results = ai.analyze_whiteboard(
    image_path,
    detect_regions=True,    # Auto-detect regions
    extract_actions=True    # Extract action items
)

# Print results
print("\n📝 TRANSCRIBED TEXT:")
print(results['full_text'])

print("\n🎯 ACTION ITEMS:")
for i, item in enumerate(results['action_items'], 1):
    print(f"\n{i}. {item.task}")
    print(f"   Assignee: {item.assignee}")
    print(f"   Deadline: {item.deadline}")
    print(f"   Priority: {item.priority}")


# =============================================================================
# Example 2: Generate Visualization
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Generate Visualization")
print("="*80)

# Create annotated image with bounding boxes
viz_path = ai.visualize_results(
    image_path, 
    results, 
    output_path="annotated_whiteboard.jpg"
)

print(f"✅ Visualization saved to: {viz_path}")


# =============================================================================
# Example 3: Generate Full Report
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Generate Full Report")
print("="*80)

# Generate formatted text report
report = ai.generate_report(results, output_path="meeting_report.txt")
print(report)


# =============================================================================
# Example 4: Process Without Region Detection
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Process Entire Image as Single Region")
print("="*80)

# Sometimes you want to treat the whole whiteboard as one region
results_single = ai.analyze_whiteboard(
    image_path,
    detect_regions=False,   # Treat whole image as one region
    extract_actions=True
)

print("\n📝 Full whiteboard text:")
print(results_single['full_text'])


# =============================================================================
# Example 5: OCR Only (No Action Extraction)
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: OCR Only (Skip Action Extraction)")
print("="*80)

# If you only need transcription, skip the LLM step
results_ocr = ai.analyze_whiteboard(
    image_path,
    detect_regions=True,
    extract_actions=False   # Skip action item extraction
)

print("\n📝 Transcribed text from all regions:")
for i, region in enumerate(results_ocr['regions'], 1):
    print(f"\nRegion {i}: {region.text}")


# =============================================================================
# Example 6: Custom Processing Pipeline
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 6: Custom Processing (Advanced)")
print("="*80)

from PIL import Image

# Load image
image = Image.open(image_path)

# Step 1: Detect regions manually
regions = ai.detect_regions(image_path, conf_threshold=0.3)
print(f"\n🔍 Detected {len(regions)} regions")

# Step 2: OCR specific regions
for i, region in enumerate(regions[:3]):  # Only process first 3 regions
    text = ai.ocr_region(image, region)
    region.text = text
    print(f"\nRegion {i+1} text: {text}")

# Step 3: Extract action items from combined text
combined_text = "\n\n".join([r.text for r in regions])
action_items = ai.extract_action_items(combined_text)

print(f"\n🎯 Found {len(action_items)} action items")
for item in action_items:
    print(f"  - {item.task} ({item.assignee})")


# =============================================================================
# Example 7: Batch Processing Multiple Whiteboards
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 7: Batch Processing Multiple Images")
print("="*80)

# Process multiple whiteboard images
whiteboard_images = [
    "whiteboard1.jpg",
    "whiteboard2.jpg", 
    "whiteboard3.jpg"
]

all_action_items = []

for img_path in whiteboard_images:
    try:
        results = ai.analyze_whiteboard(img_path)
        all_action_items.extend(results['action_items'])
        print(f"✅ Processed {img_path}: {len(results['action_items'])} actions")
    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")

print(f"\n📊 Total action items from all whiteboards: {len(all_action_items)}")


# =============================================================================
# Example 8: Export to JSON
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 8: Export Results to JSON")
print("="*80)

import json
from dataclasses import asdict

# Convert results to JSON-serializable format
export_data = {
    "image_path": results['image_path'],
    "full_text": results['full_text'],
    "action_items": [asdict(item) for item in results['action_items']],
    "regions": [
        {
            "bbox": region.bbox,
            "confidence": region.confidence,
            "label": region.label,
            "text": region.text
        }
        for region in results['regions']
    ]
}

# Save to JSON
with open("whiteboard_analysis.json", "w") as f:
    json.dump(export_data, f, indent=2)

print("✅ Results exported to whiteboard_analysis.json")


# =============================================================================
# Example 9: Integration with Task Management
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 9: Export Action Items to CSV for Task Management")
print("="*80)

import csv

# Export action items to CSV for importing into Trello, Asana, etc.
with open("action_items.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Task", "Assignee", "Deadline", "Priority", "Source"])
    
    for i, item in enumerate(results['action_items']):
        writer.writerow([
            item.task,
            item.assignee,
            item.deadline,
            item.priority,
            f"Whiteboard Region {item.source_region}" if item.source_region >= 0 else "N/A"
        ])

print("✅ Action items exported to action_items.csv")
print("\nYou can now import this CSV into your task management tool!")


# =============================================================================
# Memory Usage Information
# =============================================================================

print("\n" + "="*80)
print("GPU MEMORY USAGE")
print("="*80)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("CUDA not available, running on CPU")


print("\n" + "="*80)
print("✅ All examples completed!")
print("="*80)
