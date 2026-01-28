from PIL import Image
import numpy as np
import os

# Test on IAM handwriting samples (what TrOCR was trained on)
iam_test_dir = "datasets/iam/test/images"

# Find a few test images
test_images = []
if os.path.exists(iam_test_dir):
    for f in os.listdir(iam_test_dir)[:5]:  # First 5 images
        if f.endswith(('.png', '.jpg')):
            test_images.append(os.path.join(iam_test_dir, f))

if not test_images:
    print("No IAM test images found. Using whiteboard sample instead.")
    test_images = ["datasets/pretest/sample_whiteboard_6.jpg"]

print("="*60)
print("TESTING OCR BACKENDS")
print("="*60)

# 1. Test EasyOCR
print("\n--- Loading EasyOCR ---")
import easyocr
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# 2. Test Hybrid (loads TrOCR + LoRA)
print("\n--- Loading Hybrid (EasyOCR + TrOCR+LoRA) ---")
from whiteboard_ai import WhiteboardAI
ai = WhiteboardAI(device='cuda', use_trained_models=True, ocr_backend='hybrid')

# Test each image
for img_path in test_images:
    print(f"\n{'='*60}")
    print(f"Image: {os.path.basename(img_path)}")
    print("="*60)
    
    image = Image.open(img_path).convert('RGB')
    print(f"Size: {image.size}")
    
    # Create fake region for full image
    class FakeRegion:
        def __init__(self, w, h):
            self.bbox = (0, 0, w, h)
    
    region = FakeRegion(image.width, image.height)
    
    # EasyOCR
    results = reader.readtext(np.array(image))
    easyocr_text = " ".join([r[1] for r in results])
    print(f"\nEasyOCR: {easyocr_text[:100]}...")
    
    # Hybrid
    hybrid_text = ai.ocr_region(image, region)
    print(f"Hybrid:  {hybrid_text[:100]}...")

print("\n" + "="*60)
print("DONE")
print("="*60)