"""
Test the fine-tuned code review model.
"""

from unsloth import FastLanguageModel

MODEL_PATH = "./output/merged_model_v2"  # Use latest checkpoint
MAX_SEQ_LENGTH = 1536

# Test code samples
TEST_SAMPLES = [
    {
        "filename": "user_service.py",
        "code": '''def get_user(user_id):
    user = db.query(User).filter(id=user_id).first()
    return user.name
'''
    },
    {
        "filename": "api.py",
        "code": '''def fetch_data(url):
    response = requests.get(url)
    return response.json()
'''
    },
    {
        "filename": "utils.py",
        "code": '''def parse_config(config_file):
    with open(config_file) as f:
        data = json.load(f)
    return data["settings"]["database"]["host"]
'''
    },
]


def review_code(model, tokenizer, filename: str, code: str) -> str:
    """Generate a code review for the given code."""
    
    prompt = f"""<|im_start|>system
You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback. Focus on bugs, potential issues, code quality, and improvements. Be direct and actionable.<|im_end|>
<|im_start|>user
Review this Python code from `{filename}`:

```python
{code}
```<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    print("\n" + "=" * 60)
    print("CODE REVIEW CRITIC - TEST")
    print("=" * 60)
    
    for i, sample in enumerate(TEST_SAMPLES, 1):
        print(f"\n--- Test {i}: {sample['filename']} ---")
        print(f"Code:\n{sample['code']}")
        print("\nReview:")
        review = review_code(model, tokenizer, sample["filename"], sample["code"])
        print(review)
        print()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Paste code (end with 'END' on a new line), or 'quit' to exit")
    print("=" * 60)
    
    while True:
        print("\nEnter code to review:")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            if line.strip().lower() == "quit":
                print("Goodbye!")
                return
            lines.append(line)
        
        if lines:
            code = "\n".join(lines)
            print("\nGenerating review...")
            review = review_code(model, tokenizer, "input.py", code)
            print(f"\nReview:\n{review}")


if __name__ == "__main__":
    main()
