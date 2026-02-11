"""
Review GitHub PRs using the fine-tuned code review model.

Usage:
    python review_pr.py https://github.com/owner/repo/pull/123
"""

import re
import sys
import requests
from unsloth import FastLanguageModel

# Configuration
MODEL_PATH = "./output/merged_model_v2"
MAX_SEQ_LENGTH = 1536
GITHUB_API = "https://api.github.com"


def parse_pr_url(url: str) -> tuple:
    """Extract owner, repo, pr_number from GitHub PR URL."""
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL: {url}")
    return match.group(1), match.group(2), int(match.group(3))


def fetch_pr_info(owner: str, repo: str, pr_number: int) -> dict:
    """Fetch PR metadata from GitHub API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
    response.raise_for_status()
    return response.json()


def fetch_pr_files(owner: str, repo: str, pr_number: int) -> list:
    """Fetch changed files from GitHub API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files"
    response = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
    response.raise_for_status()
    return response.json()


def extract_added_lines(patch: str) -> str:
    """Extract only added lines from a patch, removing diff markers."""
    added_lines = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])  # Remove the + prefix
    return "\n".join(added_lines)


def review_code(model, tokenizer, filename: str, code: str) -> str:
    """Generate a code review for the given code."""

    # Truncate code if too long
    if len(code) > 3000:
        code = code[:3000] + "\n... (truncated)"

    prompt = f"""<|im_start|>system
You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback. Focus on bugs, potential issues, code quality, and improvements. Be direct and actionable. Never say "I'm not sure" or "I think". Always provide specific, concrete feedback.<|im_end|>
<|im_start|>user
Review this Python code from `{filename}`:

```python
{code}
```<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.4,
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
    if len(sys.argv) < 2:
        print("Usage: python review_pr.py <github_pr_url>")
        print("Example: python review_pr.py https://github.com/HKUDS/nanobot/pull/109")
        sys.exit(1)

    pr_url = sys.argv[1]

    # Parse the PR URL
    try:
        owner, repo, pr_number = parse_pr_url(pr_url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 60)
    print("CODE REVIEW CRITIC - PR REVIEW")
    print("=" * 60)
    print(f"Repository: {owner}/{repo}")
    print(f"PR Number: #{pr_number}")
    print()

    # Fetch PR info
    print("Fetching PR information...")
    try:
        pr_info = fetch_pr_info(owner, repo, pr_number)
        print(f"Title: {pr_info['title']}")
        print(f"Author: {pr_info['user']['login']}")
        print(f"Status: {pr_info['state']}")
        print()
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching PR: {e}")
        sys.exit(1)

    # Fetch changed files
    print("Fetching changed files...")
    try:
        files = fetch_pr_files(owner, repo, pr_number)
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching files: {e}")
        sys.exit(1)

    # Filter Python files
    python_files = [f for f in files if f["filename"].endswith(".py")]

    if not python_files:
        print("No Python files found in this PR.")
        sys.exit(0)

    print(f"Found {len(python_files)} Python file(s) to review:")
    for f in python_files:
        print(f"  - {f['filename']} (+{f['additions']} -{f['deletions']})")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.\n")

    # Review each Python file
    for i, file_info in enumerate(python_files, 1):
        filename = file_info["filename"]
        patch = file_info.get("patch", "")
        status = file_info.get("status", "modified")

        print("=" * 60)
        print(f"[{i}/{len(python_files)}] {filename}")
        print(f"Status: {status} | +{file_info['additions']} -{file_info['deletions']}")
        print("=" * 60)

        if not patch:
            print("(No patch available - file may be binary or too large)")
            continue

        # Extract only added lines for cleaner model input
        code_to_review = extract_added_lines(patch)

        if not code_to_review.strip():
            print("(No additions in this file)")
            continue

        # Show code snippet
        print("\nAdded code:")
        print("-" * 40)
        snippet = code_to_review[:500] + ("..." if len(code_to_review) > 500 else "")
        print(snippet)
        print("-" * 40)

        # Generate review
        print("\nGenerating review...")
        review = review_code(model, tokenizer, filename, code_to_review)

        print("\nReview:")
        print(review)
        print()

    print("=" * 60)
    print("REVIEW COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
