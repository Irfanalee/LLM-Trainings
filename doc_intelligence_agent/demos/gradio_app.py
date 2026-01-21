"""
Document Intelligence Agent - Gradio Demo
A beautiful web interface for demonstrating the document AI capabilities

This creates an impressive demo perfect for LinkedIn posts and showcasing!
"""

import gradio as gr
import json
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Global agent instance (loaded once)
agent = None


def load_agent():
    """Load the agent (called once on startup)"""
    global agent
    if agent is None:
        from document_agent import DocumentIntelligenceAgent
        print("Loading Document Intelligence Agent...")
        agent = DocumentIntelligenceAgent()
    return agent


def process_document(
    file,
    task: str,
    question: str,
    extraction_fields: str,
    summary_style: str
):
    """Main processing function for the Gradio interface"""
    
    if file is None:
        return "Please upload a document first.", None, None
    
    try:
        # Load agent
        doc_agent = load_agent()
        
        # Load image
        if isinstance(file, str):
            image = Image.open(file).convert("RGB")
        else:
            image = Image.open(file.name).convert("RGB")
        
        # Perform analysis
        analysis = doc_agent.analyze_document(image)
        
        # Format analysis output
        analysis_output = f"""## 📝 OCR Text Extracted
{analysis.ocr_text if analysis.ocr_text else "No text detected"}

---

## 🖼️ Document Description
{analysis.detailed_caption if analysis.detailed_caption else "No caption generated"}

---

## 📍 Detected Elements
{json.dumps(analysis.objects_detected, indent=2) if analysis.objects_detected else "No objects detected"}
"""
        
        # Task-specific processing
        if task == "Question Answering":
            if not question.strip():
                task_output = "Please enter a question to ask about the document."
            else:
                answer = doc_agent.ask_question(image, question, analysis)
                task_output = f"""## 🤔 Question
{question}

## 💡 Answer
{answer}
"""
        
        elif task == "Structured Extraction":
            # Parse extraction fields
            if not extraction_fields.strip():
                task_output = "Please specify fields to extract (one per line, format: field_name: description)"
            else:
                schema = {}
                for line in extraction_fields.strip().split("\n"):
                    if ":" in line:
                        key, desc = line.split(":", 1)
                        schema[key.strip()] = desc.strip()
                
                if schema:
                    extracted = doc_agent.extract_structured_data(image, schema, analysis)
                    task_output = f"""## 📊 Extracted Data
```json
{json.dumps(extracted, indent=2)}
```
"""
                else:
                    task_output = "Could not parse extraction fields. Use format: field_name: description"
        
        elif task == "Summarization":
            summary = doc_agent.summarize(image, summary_style.lower(), analysis)
            task_output = f"""## 📋 Summary ({summary_style})
{summary}
"""
        
        else:
            task_output = "Please select a task."
        
        return analysis_output, task_output, image
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing document: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None


def create_demo():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Document Intelligence Agent",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-title">
            <h1>🔍 Document Intelligence Agent</h1>
        </div>
        <div class="subtitle">
            <p>AI-powered document understanding using Florence-2 + Qwen2.5</p>
            <p><em>Upload any document to analyze, ask questions, or extract data</em></p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Upload Document")
                
                file_input = gr.File(
                    label="Upload PDF or Image",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
                    type="filepath"
                )
                
                image_preview = gr.Image(
                    label="Document Preview",
                    type="pil",
                    interactive=False
                )
                
                gr.Markdown("### ⚙️ Task Selection")
                
                task_selector = gr.Radio(
                    choices=["Question Answering", "Structured Extraction", "Summarization"],
                    value="Question Answering",
                    label="What would you like to do?"
                )
                
                # Conditional inputs for each task
                with gr.Group() as qa_group:
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the total amount due?",
                        lines=2
                    )
                
                with gr.Group(visible=False) as extract_group:
                    extraction_input = gr.Textbox(
                        label="Fields to Extract (one per line)",
                        placeholder="invoice_number: The invoice ID\ndate: The document date\ntotal: The total amount",
                        lines=4
                    )
                
                with gr.Group(visible=False) as summary_group:
                    summary_style = gr.Dropdown(
                        choices=["Concise", "Detailed", "Bullet", "Executive"],
                        value="Concise",
                        label="Summary Style"
                    )
                
                process_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                analysis_output = gr.Markdown(
                    label="Document Analysis",
                    value="*Upload a document and click 'Process' to see results*"
                )
                
                gr.Markdown("### 💡 Task Results")
                
                task_output = gr.Markdown(
                    label="Task Output",
                    value="*Results will appear here*"
                )
        
        # Examples section
        gr.Markdown("---")
        gr.Markdown("### 📌 Example Use Cases")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                **📧 Invoice Processing**
                - Extract invoice numbers, dates, amounts
                - Verify line items and totals
                - Flag discrepancies
                """)
            with gr.Column():
                gr.Markdown("""
                **📋 Contract Analysis**
                - Identify key terms and dates
                - Extract party names and obligations
                - Summarize main clauses
                """)
            with gr.Column():
                gr.Markdown("""
                **📊 Technical Documents**
                - P&ID diagram analysis
                - Engineering drawing review
                - Specification extraction
                """)
        
        # Task visibility logic
        def update_task_visibility(task):
            return (
                gr.update(visible=(task == "Question Answering")),
                gr.update(visible=(task == "Structured Extraction")),
                gr.update(visible=(task == "Summarization"))
            )
        
        task_selector.change(
            fn=update_task_visibility,
            inputs=[task_selector],
            outputs=[qa_group, extract_group, summary_group]
        )
        
        # Process button click
        process_btn.click(
            fn=process_document,
            inputs=[
                file_input,
                task_selector,
                question_input,
                extraction_input,
                summary_style
            ],
            outputs=[analysis_output, task_output, image_preview]
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("Document Intelligence Agent - Demo")
    print("=" * 50)
    print()
    print("Starting Gradio interface...")
    print("This may take a moment to load models on first run.")
    print()
    
    demo = create_demo()
    
    # Launch with share=True to get a public URL for demos
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to get a public URL
        show_error=True
    )
