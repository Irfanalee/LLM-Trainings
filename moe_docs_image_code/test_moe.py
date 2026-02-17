"""
Test the Fine-tuned Document Intelligence MoE Model
"""

import torch
from unsloth import FastLanguageModel
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "output" / "lora_adapters"
MAX_SEQ_LENGTH = 2048

# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    {
        "name": "Invoice Extraction",
        "system": """You are an expert invoice and receipt analyst. Extract structured data from the document image.

Respond with JSON containing:
- vendor: Company/store name
- date: Document date
- items: List of line items with description, quantity, price
- subtotal: Amount before tax
- tax: Tax amount
- total: Final amount
- currency: Currency used

Then provide a brief summary.""",
        "user": """Extract all information from this receipt:

COFFEE HOUSE
123 Main Street
Date: 2024-02-15

Cappuccino      $4.50
Croissant       $3.25
Latte           $5.00

Subtotal:      $12.75
Tax (8%):       $1.02
Total:         $13.77

Payment: Credit Card
Thank you!"""
    },
    {
        "name": "Contract Analysis",
        "system": """You are an expert legal document analyst. Extract key information from contracts and agreements.

Respond with JSON containing:
- document_type: Type of agreement
- parties: List of parties involved
- effective_date: When agreement starts
- key_terms: Important terms and conditions
- obligations: What each party must do
- termination: How the agreement can end

Then provide a brief summary of the contract.""",
        "user": """Analyze this contract excerpt:

SERVICE AGREEMENT

This Agreement is entered into as of January 1, 2024 ("Effective Date") by and between:

Party A: TechCorp Inc., a Delaware corporation
Party B: ConsultingPro LLC, a California limited liability company

TERMS:
1. ConsultingPro shall provide software development services for 12 months.
2. TechCorp shall pay $15,000 per month for services rendered.
3. Either party may terminate with 30 days written notice.
4. All intellectual property created shall belong to TechCorp.
5. ConsultingPro shall maintain confidentiality of all proprietary information.

GOVERNING LAW: This Agreement shall be governed by the laws of the State of Delaware."""
    },
    {
        "name": "General Document QA",
        "system": """You are an expert document analyst. Answer questions about the document accurately and concisely.

Provide direct answers based on what you can see in the document. If the information is not visible, say so.""",
        "user": """From this memo:

INTERNAL MEMO
To: All Employees
From: Sarah Johnson, CEO
Date: March 15, 2024
Subject: Q1 Performance Update

I'm pleased to announce that Q1 2024 exceeded expectations:
- Revenue: $4.2M (up 15% YoY)
- New customers: 127
- Employee satisfaction: 87%

Our engineering team shipped 3 major features, and sales closed our largest deal ever with GlobalCorp ($500K ARR).

Next quarter priorities:
1. Launch mobile app
2. Expand to European market
3. Hire 10 new engineers

Questions? Join the all-hands meeting on March 20th at 2pm.

Question: What was the Q1 revenue and how much did it grow?"""
    },
]

# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    """Load the fine-tuned model."""
    print("Loading fine-tuned model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Set for inference
    FastLanguageModel.for_inference(model)
    
    print("Model loaded!")
    return model, tokenizer


def generate_response(model, tokenizer, system: str, user: str) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response (after the prompt)
    if prompt in response:
        response = response[len(prompt):].strip()
    
    return response


def run_tests(model, tokenizer):
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("RUNNING TEST CASES")
    print("=" * 60)
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        print(f"\nInput:\n{test['user'][:200]}...")
        
        response = generate_response(model, tokenizer, test["system"], test["user"])
        
        print(f"\n🤖 Response:\n{response}")
        print("-" * 40)


def interactive_mode(model, tokenizer):
    """Interactive testing mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    system_prompt = """You are an expert document analyst. Analyze documents and extract structured information.
For invoices: Extract vendor, amounts, items, dates.
For contracts: Extract parties, terms, obligations.
For general documents: Provide summaries and answer questions."""
    
    while True:
        print("\n" + "-" * 40)
        user_input = input("Enter document text or question (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        response = generate_response(model, tokenizer, system_prompt, user_input)
        print(f"\n🤖 Response:\n{response}")


def main():
    print("=" * 60)
    print("DOCUMENT INTELLIGENCE MOE - TEST")
    print("=" * 60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found at: {MODEL_PATH}")
        print("Run training first: python train_moe.py")
        return
    
    # Load model
    model, tokenizer = load_model()
    
    # Run tests
    run_tests(model, tokenizer)
    
    # Interactive mode
    interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
