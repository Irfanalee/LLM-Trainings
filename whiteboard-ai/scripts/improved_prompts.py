#!/usr/bin/env python3
"""
Improved Prompts for Better Zero-Shot Performance
Use these before investing in fine-tuning!

These prompts can significantly improve action item extraction
without any training data.
"""

# Basic prompt (current implementation)
BASIC_PROMPT = """You are an AI assistant that extracts action items from meeting notes.

Meeting Notes:
{meeting_notes}

Extract all action items and format them as JSON array. Each action item should have:
- task: what needs to be done
- assignee: who is responsible (if mentioned, otherwise "Unassigned")
- deadline: when it's due (if mentioned, otherwise "No deadline")
- priority: High/Normal/Low (infer from context)

Return ONLY the JSON array, no other text.

JSON:"""


# Improved prompt with better instructions and examples
IMPROVED_PROMPT_V1 = """You are an expert meeting analyst. Your job is to extract actionable tasks from meeting notes.

## Instructions:
1. Identify ALL action items, tasks, and to-dos mentioned
2. Look for keywords like: TODO, action item, need to, should, must, will, @person, by [date]
3. Extract the person responsible (look for names, @mentions, "assigned to")
4. Extract deadlines (dates, "by Friday", "next week", "ASAP", etc.)
5. Infer priority from urgency words (URGENT, ASAP, critical = High; eventually, when possible = Low)

## Meeting Notes:
{meeting_notes}

## Your Task:
Extract all action items as a JSON array. Be thorough - don't miss any tasks!

## Output Format:
```json
[
  {{"task": "Clear description of what needs to be done", "assignee": "Person name or Unassigned", "deadline": "Specific date/time or No deadline", "priority": "High/Normal/Low"}}
]
```

Return ONLY valid JSON, no explanations."""


# Chain-of-thought prompt for more accurate extraction
COT_PROMPT = """You are an expert meeting analyst. Extract action items step by step.

## Meeting Notes:
{meeting_notes}

## Step 1: Identify all action-related sentences
List sentences that contain tasks, to-dos, or actions.

## Step 2: For each action, extract:
- What is the task? (be specific)
- Who is responsible? (look for names, @mentions)
- When is it due? (look for dates, timeframes)
- How urgent is it? (look for urgency indicators)

## Step 3: Format as JSON
After your analysis, provide the final JSON array:

```json
[
  {{"task": "...", "assignee": "...", "deadline": "...", "priority": "..."}}
]
```

Begin your analysis:"""


# Few-shot prompt with examples
FEW_SHOT_PROMPT = """You extract action items from meeting notes. Here are examples:

### Example 1:
Input: "John will update the API docs by Friday. Sarah needs to review the PR."
Output:
```json
[
  {{"task": "Update the API docs", "assignee": "John", "deadline": "Friday", "priority": "Normal"}},
  {{"task": "Review the PR", "assignee": "Sarah", "deadline": "No deadline", "priority": "Normal"}}
]
```

### Example 2:
Input: "URGENT: Fix the login bug ASAP - @Mike. Also, we should update the README eventually."
Output:
```json
[
  {{"task": "Fix the login bug", "assignee": "Mike", "deadline": "ASAP", "priority": "High"}},
  {{"task": "Update the README", "assignee": "Unassigned", "deadline": "No deadline", "priority": "Low"}}
]
```

### Example 3:
Input: "Discussed the new feature. Need more research."
Output:
```json
[
  {{"task": "Research for new feature", "assignee": "Unassigned", "deadline": "No deadline", "priority": "Normal"}}
]
```

### Now extract from this:
Input: "{meeting_notes}"
Output:"""


# Structured output prompt (more reliable JSON)
STRUCTURED_PROMPT = """<task>Extract action items from meeting notes</task>

<meeting_notes>
{meeting_notes}
</meeting_notes>

<rules>
1. Extract ALL tasks, to-dos, and action items
2. Each item MUST have: task, assignee, deadline, priority
3. Use "Unassigned" if no person mentioned
4. Use "No deadline" if no date mentioned
5. Priority: High (urgent/ASAP), Normal (default), Low (eventually/when possible)
6. Return ONLY a JSON array, nothing else
</rules>

<output_format>
[{{"task":"string","assignee":"string","deadline":"string","priority":"High|Normal|Low"}}]
</output_format>

<json_output>"""


# Whiteboard-specific prompt (handles abbreviated text better)
WHITEBOARD_PROMPT = """You are analyzing handwritten whiteboard notes from a meeting.

## Context:
- Text may be abbreviated (e.g., "Rev docs" = "Review documents")
- Names might be shortened (e.g., "J" = "John", "S" = "Sarah")
- Dates may be informal ("Fri" = "Friday", "nxt wk" = "next week")
- Bullets or checkboxes indicate action items

## Meeting Notes (from whiteboard):
{meeting_notes}

## Instructions:
1. Expand abbreviations where obvious
2. Extract ALL action items (look for bullets, checkboxes, "TODO", "@", etc.)
3. Infer assignees from context (nearby names)
4. Normalize deadlines to readable format

## Output (JSON array only):
```json
[
  {{"task": "Full description (expand abbreviations)", "assignee": "Full name or Unassigned", "deadline": "Normalized date or No deadline", "priority": "High/Normal/Low"}}
]
```

JSON:"""


def get_best_prompt(meeting_notes: str, prompt_type: str = "improved") -> str:
    """
    Get the best prompt for action item extraction

    Args:
        meeting_notes: The meeting notes text
        prompt_type: One of "basic", "improved", "cot", "few_shot", "structured", "whiteboard"

    Returns:
        Formatted prompt string
    """
    prompts = {
        "basic": BASIC_PROMPT,
        "improved": IMPROVED_PROMPT_V1,
        "cot": COT_PROMPT,
        "few_shot": FEW_SHOT_PROMPT,
        "structured": STRUCTURED_PROMPT,
        "whiteboard": WHITEBOARD_PROMPT
    }

    prompt_template = prompts.get(prompt_type, IMPROVED_PROMPT_V1)
    return prompt_template.format(meeting_notes=meeting_notes)


def test_prompts():
    """Test different prompts with a sample meeting note"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import json
    import re

    sample_notes = """Sprint Planning Meeting

Discussion:
1. Reviewed the Q4 roadmap with product team
2. @Sarah presented the new design mockups
3. Backend API needs optimization before launch

Action Items:
- [ ] Review security audit report - @John by Friday
- TODO: Update API documentation - Sarah by next week
- URGENT: Fix authentication bug - Mike ASAP
- Research caching solutions (no owner yet)
- Schedule deployment review meeting"""

    print("="*60)
    print("Testing Different Prompts")
    print("="*60)

    # Load model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    prompt_types = ["basic", "improved", "few_shot", "structured", "whiteboard"]

    for prompt_type in prompt_types:
        print(f"\n{'='*40}")
        print(f"Testing: {prompt_type.upper()} prompt")
        print("="*40)

        prompt = get_best_prompt(sample_notes, prompt_type)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.95
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Try to parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                print(f"Successfully extracted {len(items)} items:")
                for item in items:
                    print(f"  - {item.get('task', 'N/A')} (@{item.get('assignee', 'N/A')})")
            else:
                print("No JSON found in response")
        except json.JSONDecodeError:
            print("Failed to parse JSON")

        print(f"\nRaw response:\n{response[:500]}...")


# Updated WhiteboardAI class with improved prompts
class ImprovedActionExtractor:
    """
    Drop-in replacement for action extraction with better prompts
    """

    def __init__(self, model, tokenizer, prompt_type: str = "whiteboard"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type

    def extract_action_items(self, meeting_notes: str):
        """Extract action items using improved prompts"""
        import json
        import re

        prompt = get_best_prompt(meeting_notes, self.prompt_type)

        messages = [
            {"role": "system", "content": "You are a meeting notes analyst. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,  # Lower temperature for more consistent JSON
                do_sample=True,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return []


if __name__ == "__main__":
    print("Improved Prompts for Whiteboard AI")
    print("="*60)
    print("\nAvailable prompt types:")
    print("  - basic: Simple prompt (current implementation)")
    print("  - improved: Better instructions")
    print("  - cot: Chain-of-thought reasoning")
    print("  - few_shot: Examples included")
    print("  - structured: XML-like structure")
    print("  - whiteboard: Handles abbreviated handwriting")
    print("\nRecommendation: Start with 'whiteboard' or 'few_shot' prompt")
    print("\nTo test prompts, run: python improved_prompts.py --test")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_prompts()
