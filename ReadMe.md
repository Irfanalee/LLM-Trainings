# Fine-Tuning a Local Asset Tagger: A Teacher's Guide

This guide documents the process of transforming a general-purpose AI model into a specialized "JSON Asset Tagger" for engineering data. By following this method, you achieve the OLI model of internationalization:

- **Ownership:** You own a custom model "brain" tailored to your industry.
- **Location:** Your sensitive data remains local and secure.
- **Internalization:** You develop the technical skill to build AI tools without relying on external APIs.

## Phase 1: Loading the Brain (The Base Model)

We start by downloading a pre-trained model and fitting it into our hardware memory.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    load_in_4bit = True,
)
```

### Explaining the code:

- **from_pretrained:** Fetches the core knowledge from the cloud.
- **load_in_4bit = True:** This is Quantization. Imagine the model's knowledge as a high-resolution 4K video. Your 16GB GPU is a smaller screen. By loading in 4-bit, we shrink that 4K video to 1080p. It fits in your memory while keeping the "movie" clear enough to watch.
- **tokenizer:** This is the translator. It turns your English words into lists of numbers (tokens) that the neural network can process.

## Phase 2: The Sticky Note (LoRA)

We do not retrain the entire 1.5 billion synapses. That would be too slow. Instead, we use LoRA (Low-Rank Adaptation). Think of this as attaching a "sticky note" to the brain.

```python
model = FastLanguageModel.get_peft_model(
    model, 
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

### Explaining the code:

- **r = 16 (The Rank):** This determines the "width" or "surface area" of your sticky note.
  - A lower rank (like 4 or 8) is a tiny note. It saves memory but might not learn complex patterns.
  - A higher rank (like 16 or 32) is a larger note. It gives the model more space to learn precise engineering syntax. 16 is the professional standard for specific tasks like JSON extraction.

- **target_modules:** These are the specific sections of the brain where we attach the sticky notes. These four modules handle Attention:
  - **q_proj (Query):** Helps the model ask "What am I looking for?" (e.g., "Where is the valve type?").
  - **k_proj (Key):** Helps the model label the data it finds (e.g., "This word 'gate' is the valve type").
  - **v_proj (Value):** Helps the model remember the content (e.g., "The type is 'gate'").
  - **o_proj (Output):** Ensures the final answer is structured correctly as JSON.

## Phase 3: The Flashcards (Data Prep)

The model needs structured examples to learn a new pattern.

```python
alpaca_prompt = """### Instruction: {} ### Input: {} ### Response: {}"""

def formatting_prompts_func(examples):
    # This glues our spreadsheet columns into a single wall of text.
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }
```

### Explaining the code:

- **alpaca_prompt:** This is a template. It teaches the model to recognize where your question ends and the answer begins.
- **tokenizer.eos_token:** The "End of String" marker. This is vital. It teaches the model to shut up once it finishes the JSON. Without it, the model will babble endlessly.

## Phase 4: The Workout (The Trainer)

This is the actual learning process where the model reviews the data.

```python
trainer = SFTTrainer(
    model = model,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60,
        learning_rate = 2e-4,
        report_to = "none",
    ),
)
```

### Explaining the code:

- **batch_size = 2:** The model looks at 2 flashcards at once. This keeps the 16GB GPU from overheating.
- **gradient_accumulation_steps = 4:** The model waits until it has seen 8 examples (2 x 4) before it updates its sticky note. This makes the learning more stable and accurate.
- **max_steps = 60:** We run 60 rounds of learning. For a small 20-example dataset, 60 is the "sweet spot" for learning without memorizing.
- **learning_rate = 2e-4:** The speed of learning. Too high, and the model "overwrites" its basic coding knowledge. Too low, and it never learns the JSON format.

## Phase 5: The Final Exam (Inference)

We test the new "brain" on a sentence it has never seen.

```python
FastLanguageModel.for_inference(model)
outputs = model.generate(**inputs, max_new_tokens = 64)
```

### Explaining the code:

- **for_inference:** Disables the "learning" mode and enables "speed" mode. It makes the model 2x faster.
- **max_new_tokens = 64:** Tells the model to stop writing once the JSON object is complete.

## How to Run in Google Colab

Follow these steps to run the training in the cloud:

### Set Runtime
Go to Runtime > Change runtime type and select T4 GPU.

### Setup Environment
Run the following code in a cell to install libraries and disable the login prompt:

```bash
!pip install unsloth vllm bitsandbytes peft accelerate trl
import os
os.environ["WANDB_DISABLED"] = "true"
```

### Upload Data
Click the folder icon on the left sidebar and upload your `dataset.jsonl` file.

### Execute Training
Run your code cells. If the console asks for a choice, type 3 and press Enter.

### Download Model
Once the model saves as a GGUF file, locate it in the file sidebar and select Download.

## Running the Finished Model Locally (Ollama)

Once you download your `.gguf` file to your desktop, follow these steps to use it:

### Create a Modelfile
Create a new text file named `Modelfile` (no extension) in the same folder as your GGUF file.

### Add Configuration

```
FROM ./my_asset_model-unsloth.Q4_K_M.gguf
PARAMETER temperature 0.1
SYSTEM "Extract equipment asset details from the sentence into a structured JSON object."
```

### Register with Ollama
Open your terminal and run:

```bash
ollama create asset-tagger -f Modelfile
```

### Run and Test

```bash
ollama run asset-tagger
>>> There is a red manual gate valve on line 102.
```

## Conclusion

You have now created a specialized GGUF file. This model is no longer just a coder; it is a dedicated Engineering Asset Tagger that runs offline on your 16GB desktop.