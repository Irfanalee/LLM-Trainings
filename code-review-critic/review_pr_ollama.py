"""
Review GitHub PRs using the Ollama code review model (CPU-based).

Usage:
    python review_pr_ollama.py https://github.com/owner/repo/pull/123
"""

import re
import sys
import requests

# Configuration
OLLAMA_MODEL = "code-review-critic-py"
OLLAMA_API = "http://localhost:11434/api/generate"
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
    """Extract only added lines from a patch."""
    added_lines = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])
    return "\n".join(added_lines)


def review_code(filename: str, code: str) -> str:
    """Generate a code review using Ollama."""
    if len(code) > 3000:
        code = code[:3000] + "\n... (truncated)"

    prompt = f"""Review this Python code from `{filename}`:

```python
{code}
```"""

    response = requests.post(
        OLLAMA_API,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python review_pr_ollama.py <github_pr_url>")
        sys.exit(1)

    pr_url = sys.argv[1]

    try:
        owner, repo, pr_number = parse_pr_url(pr_url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 60)
    print("CODE REVIEW CRITIC - PR REVIEW (Ollama)")
    print("=" * 60)
    print(f"Repository: {owner}/{repo}")
    print(f"PR Number: #{pr_number}")
    print()

    # Fetch PR info
    print("Fetching PR information...")
    pr_info = fetch_pr_info(owner, repo, pr_number)
    print(f"Title: {pr_info['title']}")
    print(f"Author: {pr_info['user']['login']}")
    print()

    # Fetch changed files
    print("Fetching changed files...")
    files = fetch_pr_files(owner, repo, pr_number)
    python_files = [f for f in files if f["filename"].endswith(".py")]

    if not python_files:
        print("No Python files found in this PR.")
        sys.exit(0)

    print(f"Found {len(python_files)} Python file(s) to review\n")

    # Review each file
    for i, file_info in enumerate(python_files, 1):
        filename = file_info["filename"]
        patch = file_info.get("patch", "")

        print("=" * 60)
        print(f"[{i}/{len(python_files)}] {filename}")
        print("=" * 60)

        if not patch:
            print("(No patch available)")
            continue

        code_to_review = extract_added_lines(patch)
        if not code_to_review.strip():
            print("(No additions)")
            continue

        print("\nGenerating review...")
        review = review_code(filename, code_to_review)
        print(f"\nReview:\n{review}\n")

    print("=" * 60)
    print("REVIEW COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
