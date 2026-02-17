"""
Configuration for Document Intelligence MoE
Using Qwen3-30B-A3B with Unsloth
"""

from pathlib import Path
from dataclasses import dataclass

# =============================================================================
# PATHS
# =============================================================================

BASE_PATH = Path(__file__).parent

@dataclass
class PathConfig:
    """Path configuration."""
    base: Path = BASE_PATH
    datasets: Path = BASE_PATH / "data" / "datasets"
    training: Path = BASE_PATH / "data" / "training"
    output: Path = BASE_PATH / "output"
    
    train_file: Path = BASE_PATH / "data" / "training" / "train.jsonl"
    eval_file: Path = BASE_PATH / "data" / "training" / "eval.jsonl"
    lora_path: Path = BASE_PATH / "output" / "lora_adapters"
    merged_path: Path = BASE_PATH / "output" / "merged_model"


# =============================================================================
# MODEL
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration for Qwen3-30B-A3B."""
    # Model
    model_name: str = "unsloth/Qwen3-30B-A3B"
    max_seq_length: int = 2048
    
    # Quantization
    load_in_4bit: bool = True  # Required for 16GB VRAM
    dtype: str = "bfloat16"
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 2
    batch_size: int = 1
    gradient_accumulation: int = 16  # Effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Optimizations
    optim: str = "adamw_8bit"
    bf16: bool = True
    packing: bool = True  # Unsloth packing


# =============================================================================
# DATA
# =============================================================================

@dataclass
class DataConfig:
    """Data configuration."""
    max_examples_per_dataset: int = 5000
    train_split: float = 0.9
    seed: int = 42


# =============================================================================
# DOCUMENT TYPES
# =============================================================================

DOCUMENT_TYPES = {
    "invoice": {
        "description": "Invoices, receipts, bills, payment requests",
        "dataset": "cord",
        "extract_fields": [
            "vendor", "date", "items", "subtotal", "tax", "total", "currency"
        ]
    },
    "contract": {
        "description": "Contracts, agreements, NDAs, legal documents",
        "dataset": "cuad",
        "extract_fields": [
            "document_type", "parties", "effective_date", "key_terms", 
            "obligations", "termination", "governing_law"
        ]
    },
    "general": {
        "description": "General documents, reports, memos, letters",
        "dataset": "docvqa",
        "extract_fields": [
            "summary", "key_points", "entities", "dates", "questions_answered"
        ]
    }
}


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS = {
    "default": """You are an expert document analyst. Analyze the provided document and extract structured information.

For invoices/receipts: Extract vendor, date, items, amounts, totals.
For contracts: Extract parties, terms, obligations, key clauses.
For general documents: Provide summary, key points, and relevant entities.

Always respond with:
1. **Document Type**: What kind of document this is
2. **Extracted Data**: JSON with structured fields
3. **Summary**: Brief natural language explanation""",

    "invoice": """You are an expert invoice and receipt analyst. Extract structured data from the document.

Respond with JSON containing:
- vendor: Company/store name
- date: Document date (YYYY-MM-DD)
- items: List of {description, quantity, unit_price, amount}
- subtotal: Amount before tax
- tax: Tax amount
- total: Final amount
- currency: Currency code (USD, EUR, etc.)

Then provide a brief summary.""",

    "contract": """You are an expert legal document analyst. Extract key information from contracts.

Respond with JSON containing:
- document_type: Type of agreement (NDA, Service Agreement, etc.)
- parties: List of parties with name and role
- effective_date: Start date (YYYY-MM-DD)
- expiration_date: End date if specified
- key_terms: List of important terms
- obligations: What each party must do
- termination: How agreement can be terminated
- governing_law: Jurisdiction

Then provide a brief summary and any risk flags.""",

    "general": """You are an expert document analyst. Answer questions about the document accurately.

Provide:
- Direct answer to the question
- Supporting context from the document
- Confidence level (high/medium/low)

If information is not in the document, clearly state that."""
}
