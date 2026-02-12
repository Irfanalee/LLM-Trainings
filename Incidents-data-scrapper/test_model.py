"""
Test the fine-tuned DevOps Incident Responder model.
Uses transformers directly (avoids Unsloth's tokenizer template validation issues
with merged models whose name_or_path no longer contains 'mistral').
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./output/merged_model"
MAX_SEQ_LENGTH = 2048

# Test incidents
TEST_INCIDENTS = [
    {
        "name": "Kubernetes OOMKilled",
        "tech": "kubernetes",
        "error": """kubectl describe pod api-server-7d4b8c6f5-x2k9m
State:          Terminated
Reason:         OOMKilled
Exit Code:      137
Restart Count:  5

Events:
  Warning  OOMKilled  Container exceeded memory limit (512Mi)"""
    },
    {
        "name": "Docker connection refused",
        "tech": "docker",
        "error": """docker logs app-container
Error: connect ECONNREFUSED 127.0.0.1:5432
FATAL: Connection to database failed"""
    },
    {
        "name": "Terraform state lock",
        "tech": "terraform",
        "error": """terraform apply
Error: Error acquiring the state lock

Lock Info:
  ID:        abc-123-def
  Path:      s3://my-bucket/terraform.tfstate
  Operation: OperationTypeApply
  Who:       user@host
  Created:   2024-01-15 10:30:00 UTC"""
    },
    {
        "name": "Redis max memory",
        "tech": "redis",
        "error": """redis-cli SET mykey "value"
(error) OOM command not allowed when used memory > 'maxmemory'."""
    },
    {
        "name": "PostgreSQL too many connections",
        "tech": "postgresql",
        "error": """psql -U postgres
FATAL: too many connections for role "postgres"
DETAIL: max_connections is 100 but 100 connections are already open."""
    },
]

SYSTEM_PROMPT = """You are an expert DevOps engineer and SRE. Analyze the provided error logs, stack traces, or incident descriptions.

Your response should include:
1. **Root Cause**: What is causing this issue
2. **Severity**: Low / Medium / High / Critical
3. **Fix**: Step-by-step solution to resolve the issue
4. **Prevention**: How to prevent this in the future (optional)

Be direct, specific, and actionable. Reference exact commands, config changes, or code fixes when applicable."""


CHAT_TEMPLATE = (
    "{{'<extra_id_0>System'}}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{'\n' + message['content'].strip()}}"
    "{% endif %}"
    "{% endfor %}"
    "{{'\n'}}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '\n<extra_id_1>User\n' + message['content'].strip() + '\n<extra_id_1>Assistant\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'].strip() }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{% endif %}"
)


def load_model():
    print("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.chat_template = CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, tech: str, error: str) -> str:
    """Generate incident response."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this {tech} incident and provide diagnosis and fix:\n\n```\n{error}\n```"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response (NeMo uses <extra_id_1>Assistant marker)
    if "<extra_id_1>Assistant" in response:
        response = response.split("<extra_id_1>Assistant")[-1].strip()
    
    return response


def main():
    model, tokenizer = load_model()
    
    print("=" * 60)
    print("DEVOPS INCIDENT RESPONDER - TEST")
    print("=" * 60)
    
    for test in TEST_INCIDENTS:
        print(f"\n--- {test['name']} ---")
        print(f"Tech: {test['tech']}")
        print(f"\nError:\n{test['error'][:200]}...")
        
        response = generate_response(model, tokenizer, test['tech'], test['error'])
        
        print(f"\n🤖 Response:\n{response}")
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Paste error logs (end with 'END' on a new line), or 'quit' to exit")
    print("=" * 60)
    
    while True:
        print("\nTech (kubernetes/docker/terraform/etc): ", end="")
        tech = input().strip()
        
        if tech.lower() == 'quit':
            break
        
        print("Paste error logs (end with 'END'):")
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        
        error = '\n'.join(lines)
        
        if error:
            response = generate_response(model, tokenizer, tech, error)
            print(f"\n🤖 Response:\n{response}")


if __name__ == "__main__":
    main()
