"""
Whiteboard Meeting Notes AI
Detects regions, OCR handwriting, extracts action items
Optimized for RTX A4000 16GB GPU
"""

import torch
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import cv2
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
import re
import os
from peft import PeftModel, LoraConfig
from scripts.improved_prompts import get_best_prompt, WHITEBOARD_PROMPT

@dataclass
class Region:
    """Detected whiteboard region"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    label: str
    text: str = ""
    
@dataclass
class ActionItem:
    """Extracted action item"""
    task: str
    assignee: str = "Unassigned"
    deadline: str = "No deadline"
    priority: str = "Normal"
    source_region: int = -1


class WhiteboardAI:
    """Complete whiteboard analysis pipeline"""

    def __init__(self, device="cuda", use_trained_models=True):
        self.device = device
        print(f"🚀 Initializing Whiteboard AI on {device}...")

        # Get the base directory for model paths
        # whiteboard_ai.py is in whiteboard-ai/, so parent is LLM-Trainings/
        script_dir = os.path.dirname(os.path.abspath(__file__))  # whiteboard-ai/
        base_dir = os.path.dirname(script_dir)  # LLM-Trainings/

        # 1. Region Detection Model (YOLOv8)
        print("📦 Loading region detection model...")
        if use_trained_models:
            # Use your trained whiteboard detector
            yolo_path = os.path.join(base_dir, "runs/detect/runs/whiteboard_yolo/yolo_whiteboard_n/weights/best.pt")
            if os.path.exists(yolo_path):
                print(f"   Using trained model: {yolo_path}")
                self.region_detector = YOLO(yolo_path)
            else:
                print(f"   ⚠️ Trained model not found, using pretrained yolov8n.pt")
                self.region_detector = YOLO('yolov8n.pt')
        else:
            self.region_detector = YOLO('yolov8n.pt')

        # 2. Handwriting OCR Model (TrOCR + LoRA)
        print("✍️  Loading handwriting OCR model...")
        self.ocr_processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-large-handwritten'
        )
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-large-handwritten'
        ).to(device)

        # Load TrOCR LoRA adapter if available
        if use_trained_models:
            trocr_lora_path = os.path.join(script_dir, "runs/trocr_handwriting/lora_adapter/adapter_weights.pt")
            if os.path.exists(trocr_lora_path):
                print(f"   Loading TrOCR LoRA adapter...")
                # Load the LoRA weights we saved manually
                adapter_state = torch.load(trocr_lora_path, map_location=device)
                # Update model with LoRA weights
                model_state = self.ocr_model.state_dict()
                for key, value in adapter_state.items():
                    if key in model_state:
                        model_state[key] = value
                self.ocr_model.load_state_dict(model_state)
                print(f"   ✅ Loaded {len(adapter_state)} LoRA weights")
            else:
                print(f"   ⚠️ TrOCR LoRA not found, using base model")

        # 3. Language Model for Action Item Extraction (Qwen2.5-7B + LoRA)
        print("🧠 Loading language model for action extraction...")

        # 4-bit quantization config for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load Qwen LoRA adapter if available
        if use_trained_models:
            qwen_lora_path = os.path.join(script_dir, "runs/qwen_action_items/lora_adapter")
            if os.path.exists(qwen_lora_path):
                print(f"   Loading Qwen LoRA adapter...")
                self.llm = PeftModel.from_pretrained(self.llm, qwen_lora_path)
                print(f"   ✅ Qwen LoRA adapter loaded")
            else:
                print(f"   ⚠️ Qwen LoRA not found, using base model")

        print("✅ All models loaded successfully!")
        
    def detect_regions(self, image_path: str, conf_threshold: float = 0.25) -> List[Region]:
        """
        Detect whiteboard regions using YOLOv8
        Returns regions sorted top-to-bottom, left-to-right
        """
        print(f"\n🔍 Detecting regions in {image_path}...")
        
        # Run YOLO detection
        results = self.region_detector(image_path, conf=conf_threshold)
        
        regions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                regions.append(Region(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    label=label
                ))
        
        # Sort regions: top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        print(f"✅ Found {len(regions)} regions")
        return regions
    
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess whiteboard region for better OCR
        - Convert to grayscale
        - Increase contrast
        - Denoise
        """
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Binarization (Otsu's method)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        return Image.fromarray(binary).convert('RGB')
    
    def ocr_region(self, image: Image.Image, region: Region) -> str:
        """
        Perform OCR on a detected region
        """
        # Crop region from image
        x1, y1, x2, y2 = region.bbox
        cropped = image.crop((x1, y1, x2, y2))
        
        # Preprocess for better OCR
        preprocessed = self.preprocess_for_ocr(cropped)
        
        # Run TrOCR
        pixel_values = self.ocr_processor(
            images=preprocessed, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.ocr_model.generate(pixel_values)
        
        text = self.ocr_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return text.strip()
    
    def extract_action_items(self, meeting_notes: str) -> List[ActionItem]:
        """
        Extract structured action items using LLM
        """
        print("\n🎯 Extracting action items...")
        prompt = get_best_prompt(meeting_notes)

#         prompt = f"""You are an AI assistant that extracts action items from meeting notes.
#  Meeting Notes:
#  {meeting_notes}
#  Extract all action items and format them as JSON array. Each action item should have:
#  - task: what needs to be done
#  - assignee: who is responsible (if mentioned, otherwise "Unassigned")
#  - deadline: when it's due (if mentioned, otherwise "No deadline")
#  - priority: High/Normal/Low (infer from context)
#  Return ONLY the JSON array, no other text.
#  Example format:
#  [
#    {{"task": "Review Q4 budget", "assignee": "Sarah", "deadline": "Friday", "priority": "High"}},
#    {{"task": "Update documentation", "assignee": "Unassigned", "deadline": "No deadline", "priority": "Normal"}}
#  ]
#  JSON:"""

        messages = [
            {"role": "system", "content": "You are a helpful meeting notes assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                action_items_data = json.loads(json_match.group())
                action_items = [
                    ActionItem(**item) for item in action_items_data
                ]
            else:
                action_items = []
        except json.JSONDecodeError:
            print("⚠️  Failed to parse action items, returning empty list")
            action_items = []
        
        print(f"✅ Extracted {len(action_items)} action items")
        return action_items
    
    def analyze_whiteboard(
        self, 
        image_path: str,
        detect_regions: bool = True,
        extract_actions: bool = True
    ) -> Dict:
        """
        Complete whiteboard analysis pipeline
        
        Args:
            image_path: Path to whiteboard image
            detect_regions: Whether to use region detection (True) or treat whole image as one region (False)
            extract_actions: Whether to extract action items from OCR text
            
        Returns:
            Dictionary with regions, OCR text, and action items
        """
        print(f"\n{'='*60}")
        print(f"📋 ANALYZING WHITEBOARD: {image_path}")
        print(f"{'='*60}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect regions or use whole image
        if detect_regions:
            regions = self.detect_regions(image_path)
            # Fallback: if no regions detected, process whole image
            if not regions:
                print("⚠️  No regions detected, falling back to full image OCR...")
                w, h = image.size
                regions = [Region(
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    label="full_image"
                )]
        else:
            # Treat whole image as one region
            w, h = image.size
            regions = [Region(
                bbox=(0, 0, w, h),
                confidence=1.0,
                label="whiteboard"
            )]
        
        # OCR each region
        print(f"\n✍️  Performing OCR on {len(regions)} regions...")
        all_text = []
        for i, region in enumerate(regions):
            text = self.ocr_region(image, region)
            region.text = text
            all_text.append(f"Region {i+1}: {text}")
            print(f"  Region {i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Combine all text
        meeting_notes = "\n\n".join(all_text)
        
        # Extract action items
        action_items = []
        if extract_actions and meeting_notes.strip():
            action_items = self.extract_action_items(meeting_notes)
        
        return {
            "image_path": image_path,
            "regions": regions,
            "full_text": meeting_notes,
            "action_items": action_items
        }
    
    def visualize_results(
        self, 
        image_path: str, 
        results: Dict, 
        output_path: str = "whiteboard_analysis.jpg"
    ):
        """
        Visualize detection results with bounding boxes and text
        """
        print(f"\n🎨 Creating visualization...")
        
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw regions
        for i, region in enumerate(results['regions']):
            x1, y1, x2, y2 = region.bbox
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw region number
            draw.rectangle([x1, y1-25, x1+60, y1], fill="red")
            draw.text((x1+5, y1-22), f"Region {i+1}", fill="white", font=small_font)
        
        image.save(output_path)
        print(f"✅ Visualization saved to {output_path}")
        return output_path
    
    def generate_report(self, results: Dict, output_path: str = "meeting_report.txt"):
        """
        Generate a formatted meeting report
        """
        print(f"\n📄 Generating report...")
        
        report = []
        report.append("=" * 80)
        report.append("WHITEBOARD MEETING NOTES - AI ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # OCR Results
        report.append("📝 TRANSCRIBED NOTES")
        report.append("-" * 80)
        for i, region in enumerate(results['regions']):
            report.append(f"\nRegion {i+1} ({region.label}):")
            report.append(f"  {region.text}")
        report.append("")
        
        # Action Items
        if results['action_items']:
            report.append("")
            report.append("🎯 ACTION ITEMS")
            report.append("-" * 80)
            for i, item in enumerate(results['action_items'], 1):
                report.append(f"\n{i}. {item.task}")
                report.append(f"   👤 Assignee: {item.assignee}")
                report.append(f"   📅 Deadline: {item.deadline}")
                report.append(f"   ⚡ Priority: {item.priority}")
        else:
            report.append("\n⚠️  No action items detected")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Report saved to {output_path}")
        return report_text


def main():
    """Demo the whiteboard AI"""
    
    # Initialize the AI
    ai = WhiteboardAI(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Example usage
    print("\n" + "="*80)
    print("WHITEBOARD AI - READY FOR ANALYSIS")
    print("="*80)
    print("\nUsage examples:")
    print("1. ai.analyze_whiteboard('path/to/whiteboard.jpg')")
    print("2. ai.visualize_results('path/to/whiteboard.jpg', results)")
    print("3. ai.generate_report(results)")
    print("\nThe AI will:")
    print("  ✓ Detect regions on the whiteboard")
    print("  ✓ OCR handwritten text using TrOCR")
    print("  ✓ Extract action items with assignees and deadlines")
    print("  ✓ Generate formatted reports")
    
    return ai


if __name__ == "__main__":
    ai = main()
