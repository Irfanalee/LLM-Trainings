# test_trocr.py (corrected)
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

# 1. Load base model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')


# 2. Apply LoRA structure first (same config as training)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)


# 3. Load your trained LoRA weights
lora_weights = torch.load('runs/trocr_handwriting/lora_adapter/adapter_weights.pt')
model.load_state_dict(lora_weights, strict=False)
model.eval().cuda()

# 4. Test on an image
image = Image.open('datasets/iam_processed/val/images/iam_000000.png').convert('RGB')
pixels = processor(images=image, return_tensors='pt').pixel_values.cuda()

# 5. Predict
with torch.no_grad():
    output = model.generate(pixel_values=pixels)
    
text = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"Predicted: {text}")
