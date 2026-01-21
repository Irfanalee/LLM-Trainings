"""
Document Intelligence Pipeline
Combines Florence-2 for vision understanding with Qwen2.5 for reasoning

This is the core engine that:
1. Processes documents (PDFs, images, scans)
2. Extracts visual and textual information
3. Answers questions and generates insights
"""

import os
import torch
from PIL import Image
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DocumentAnalysis:
    """Container for document analysis results"""
    ocr_text: str
    detailed_caption: str
    objects_detected: List[Dict]
    regions_of_interest: List[Dict]
    raw_response: str


class DocumentIntelligenceAgent:
    """
    Main agent class that orchestrates document understanding
    
    Architecture:
    - Florence-2: Handles all visual tasks (OCR, detection, captioning)
    - Qwen2.5: Handles reasoning, Q&A, and structured output generation
    """
    
    def __init__(
        self,
        florence_model: str = "microsoft/Florence-2-large",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        use_4bit: bool = True,
        device: str = "cuda"
    ):
        self.device = device
        self.use_4bit = use_4bit
        
        print("Initializing Document Intelligence Agent...")
        print(f"  Device: {device}")
        print(f"  4-bit quantization: {use_4bit}")
        
        # Load Florence-2 for vision
        self._load_florence(florence_model)
        
        # Load Qwen for reasoning
        self._load_llm(llm_model)
        
        print("\n✓ Agent initialized successfully!")
        self._print_memory_usage()
    
    def _load_florence(self, model_name: str):
        """Load Florence-2 vision-language model"""
        print(f"\n[1/2] Loading Florence-2...")
        
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        self.florence_processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        self.florence_model.eval()
        print(f"  ✓ Florence-2 loaded ({sum(p.numel() for p in self.florence_model.parameters()) / 1e6:.0f}M params)")
    
    def _load_llm(self, model_name: str):
        """Load Qwen LLM for reasoning"""
        print(f"\n[2/2] Loading Qwen2.5...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        self.llm_model.eval()
        print(f"  ✓ Qwen2.5 loaded")
    
    def _print_memory_usage(self):
        """Print current GPU memory usage"""
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Memory: {allocated:.1f}GB used / {total:.1f}GB total")
    
    def _run_florence(self, image: Image.Image, task: str, text_input: str = "") -> str:
        """Run Florence-2 on an image with a specific task"""
        
        # Florence-2 task prompts
        task_prompts = {
            "ocr": "<OCR>",
            "ocr_with_regions": "<OCR_WITH_REGION>",
            "caption": "<CAPTION>",
            "detailed_caption": "<DETAILED_CAPTION>",
            "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
            "object_detection": "<OD>",
            "dense_region_caption": "<DENSE_REGION_CAPTION>",
            "region_proposal": "<REGION_PROPOSAL>",
            "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
            "referring_expression": "<REFERRING_EXPRESSION_SEGMENTATION>",
            "open_vocab_detection": "<OPEN_VOCABULARY_DETECTION>",
        }
        
        prompt = task_prompts.get(task, task)
        if text_input:
            prompt = f"{prompt}{text_input}"
        
        inputs = self.florence_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.float16)
        
        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # Parse the response
        parsed = self.florence_processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        
        return parsed
    
    def _run_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """Run Qwen LLM for reasoning"""
        
        messages = [
            {"role": "system", "content": "You are a helpful document analysis assistant. Provide clear, structured, and accurate responses based on the document information provided."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.llm_tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        response = self.llm_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def load_document(self, path: Union[str, Path]) -> List[Image.Image]:
        """Load a document (PDF or image) and return as list of images"""
        path = Path(path)
        
        if path.suffix.lower() == '.pdf':
            # Convert PDF to images
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render at 150 DPI for good quality
                pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            doc.close()
            return images
        else:
            # Load as image
            return [Image.open(path).convert("RGB")]
    
    def analyze_document(self, image: Image.Image) -> DocumentAnalysis:
        """Perform comprehensive analysis of a document page"""
        
        print("  Running OCR...")
        ocr_result = self._run_florence(image, "ocr")
        
        print("  Generating detailed caption...")
        caption_result = self._run_florence(image, "detailed_caption")
        
        print("  Detecting objects and regions...")
        od_result = self._run_florence(image, "object_detection")
        
        print("  Analyzing regions of interest...")
        region_result = self._run_florence(image, "dense_region_caption")
        
        return DocumentAnalysis(
            ocr_text=ocr_result.get("<OCR>", ""),
            detailed_caption=caption_result.get("<DETAILED_CAPTION>", ""),
            objects_detected=od_result.get("<OD>", {}),
            regions_of_interest=region_result.get("<DENSE_REGION_CAPTION>", {}),
            raw_response=str({
                "ocr": ocr_result,
                "caption": caption_result,
                "objects": od_result,
                "regions": region_result
            })
        )
    
    def ask_question(
        self,
        image: Image.Image,
        question: str,
        analysis: Optional[DocumentAnalysis] = None
    ) -> str:
        """Ask a question about a document"""
        
        # Get or create analysis
        if analysis is None:
            print("Analyzing document...")
            analysis = self.analyze_document(image)
        
        # Build context for LLM
        context = f"""I have analyzed a document. Here is the extracted information:

## OCR Text (all text found in document):
{analysis.ocr_text}

## Document Description:
{analysis.detailed_caption}

## Detected Elements:
{json.dumps(analysis.objects_detected, indent=2) if analysis.objects_detected else "None detected"}

## Regions of Interest:
{json.dumps(analysis.regions_of_interest, indent=2) if analysis.regions_of_interest else "None detected"}

---

Based on this document analysis, please answer the following question:
{question}

Provide a clear, helpful answer based only on the information extracted from the document."""

        print("Generating response...")
        response = self._run_llm(context)
        
        return response
    
    def extract_structured_data(
        self,
        image: Image.Image,
        schema: Dict[str, str],
        analysis: Optional[DocumentAnalysis] = None
    ) -> Dict[str, Any]:
        """Extract structured data according to a schema"""
        
        if analysis is None:
            print("Analyzing document...")
            analysis = self.analyze_document(image)
        
        schema_description = "\n".join([
            f"- {key}: {description}" for key, description in schema.items()
        ])
        
        prompt = f"""I have analyzed a document. Here is the extracted information:

## OCR Text:
{analysis.ocr_text}

## Document Description:
{analysis.detailed_caption}

---

Please extract the following information from this document and return it as a JSON object:

{schema_description}

Return ONLY a valid JSON object with these fields. If a field cannot be found, use null.
Do not include any explanation, just the JSON."""

        print("Extracting structured data...")
        response = self._run_llm(prompt, max_tokens=512)
        
        # Try to parse JSON from response
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {"raw_response": response, "error": "Could not parse JSON"}
    
    def summarize(
        self,
        image: Image.Image,
        style: str = "concise",
        analysis: Optional[DocumentAnalysis] = None
    ) -> str:
        """Generate a summary of the document"""
        
        if analysis is None:
            print("Analyzing document...")
            analysis = self.analyze_document(image)
        
        style_instructions = {
            "concise": "Provide a brief 2-3 sentence summary.",
            "detailed": "Provide a comprehensive summary covering all main points.",
            "bullet": "Provide a summary as bullet points of key information.",
            "executive": "Provide an executive summary suitable for business stakeholders."
        }
        
        prompt = f"""I have analyzed a document. Here is the extracted information:

## OCR Text:
{analysis.ocr_text}

## Document Description:
{analysis.detailed_caption}

---

{style_instructions.get(style, style_instructions['concise'])}"""

        print(f"Generating {style} summary...")
        return self._run_llm(prompt)


# Convenience function for quick testing
def quick_test():
    """Quick test with a sample image"""
    print("=" * 50)
    print("Document Intelligence Agent - Quick Test")
    print("=" * 50)
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create test document image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.text((50, 30), "INVOICE #12345", fill='black', font=font)
    draw.text((50, 80), "Date: January 15, 2025", fill='black', font=small_font)
    draw.text((50, 110), "Customer: Acme Corporation", fill='black', font=small_font)
    draw.text((50, 160), "Items:", fill='black', font=small_font)
    draw.text((70, 190), "- Widget A (x10): $500.00", fill='black', font=small_font)
    draw.text((70, 220), "- Widget B (x5): $250.00", fill='black', font=small_font)
    draw.text((70, 250), "- Service Fee: $75.00", fill='black', font=small_font)
    draw.text((50, 300), "Total: $825.00", fill='black', font=font)
    draw.text((50, 360), "Payment Due: February 15, 2025", fill='black', font=small_font)
    
    # Save test image
    test_path = Path("data/test_invoice.png")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(test_path)
    print(f"\nCreated test invoice at: {test_path}")
    
    # Initialize agent
    agent = DocumentIntelligenceAgent()
    
    # Test analysis
    print("\n" + "-" * 50)
    print("Testing document analysis...")
    print("-" * 50)
    
    analysis = agent.analyze_document(img)
    print(f"\nOCR Text:\n{analysis.ocr_text[:500]}...")
    print(f"\nCaption:\n{analysis.detailed_caption}")
    
    # Test Q&A
    print("\n" + "-" * 50)
    print("Testing Q&A...")
    print("-" * 50)
    
    question = "What is the total amount due and when is it due?"
    print(f"\nQuestion: {question}")
    answer = agent.ask_question(img, question, analysis)
    print(f"\nAnswer: {answer}")
    
    # Test structured extraction
    print("\n" + "-" * 50)
    print("Testing structured extraction...")
    print("-" * 50)
    
    schema = {
        "invoice_number": "The invoice identifier",
        "date": "The invoice date",
        "customer_name": "Name of the customer",
        "total_amount": "The total amount due",
        "due_date": "Payment due date"
    }
    
    extracted = agent.extract_structured_data(img, schema, analysis)
    print(f"\nExtracted data:\n{json.dumps(extracted, indent=2)}")
    
    print("\n" + "=" * 50)
    print("Quick test complete!")
    print("=" * 50)


if __name__ == "__main__":
    quick_test()
