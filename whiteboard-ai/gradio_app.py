"""
Gradio Web Interface for Whiteboard Meeting Notes AI
Beautiful UI with drag-and-drop upload and interactive results
"""

import gradio as gr
from whiteboard_ai import WhiteboardAI, ActionItem
import torch
import json
from PIL import Image
import os

# Initialize the AI (global to avoid reloading)
print("🚀 Initializing Whiteboard AI...")
ai = WhiteboardAI(device="cuda" if torch.cuda.is_available() else "cpu")
print("✅ Ready!")


def analyze_whiteboard_image(image, use_region_detection, extract_actions):
    """
    Process whiteboard image and return results
    
    Args:
        image: PIL Image or path
        use_region_detection: bool
        extract_actions: bool
    
    Returns:
        Tuple of (annotated_image, ocr_text, action_items_html)
    """
    if image is None:
        return None, "Please upload an image first!", ""
    
    # Save temporary image
    temp_path = "/tmp/whiteboard_temp.jpg"
    if isinstance(image, str):
        temp_path = image
    else:
        image.save(temp_path)
    
    # Analyze
    results = ai.analyze_whiteboard(
        temp_path,
        detect_regions=use_region_detection,
        extract_actions=extract_actions
    )
    
    # Create visualization
    viz_path = "/tmp/whiteboard_annotated.jpg"
    ai.visualize_results(temp_path, results, viz_path)
    
    # Format OCR text
    ocr_text = results['full_text']
    
    # Format action items as HTML
    if results['action_items']:
        action_items_html = "<div style='font-family: Arial, sans-serif;'>"
        action_items_html += "<h3>🎯 Action Items Detected:</h3>"
        
        for i, item in enumerate(results['action_items'], 1):
            priority_color = {
                'High': '#ff4444',
                'Normal': '#4444ff',
                'Low': '#44ff44'
            }.get(item.priority, '#888888')
            
            action_items_html += f"""
            <div style='border-left: 4px solid {priority_color}; 
                        padding: 12px; 
                        margin: 10px 0; 
                        background-color: #f5f5f5; 
                        border-radius: 4px;'>
                <div style='font-size: 16px; font-weight: bold; margin-bottom: 8px;'>
                    {i}. {item.task}
                </div>
                <div style='font-size: 14px; color: #555;'>
                    <span style='margin-right: 15px;'>👤 <b>Assignee:</b> {item.assignee}</span>
                    <span style='margin-right: 15px;'>📅 <b>Deadline:</b> {item.deadline}</span>
                    <span style='background-color: {priority_color}; 
                                color: white; 
                                padding: 2px 8px; 
                                border-radius: 3px; 
                                font-size: 12px;'>
                        {item.priority}
                    </span>
                </div>
            </div>
            """
        action_items_html += "</div>"
    else:
        action_items_html = "<p style='color: #888;'>No action items detected. The text may not contain actionable tasks.</p>"
    
    return Image.open(viz_path), ocr_text, action_items_html


def download_report(image, use_region_detection, extract_actions):
    """
    Generate downloadable report
    """
    if image is None:
        return None
    
    # Save temporary image
    temp_path = "/tmp/whiteboard_temp.jpg"
    if isinstance(image, str):
        temp_path = image
    else:
        image.save(temp_path)
    
    # Analyze
    results = ai.analyze_whiteboard(
        temp_path,
        detect_regions=use_region_detection,
        extract_actions=extract_actions
    )
    
    # Generate report
    report_path = "/tmp/meeting_report.txt"
    ai.generate_report(results, report_path)
    
    return report_path


# Create Gradio interface
with gr.Blocks(
    title="Whiteboard Meeting Notes AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )
) as demo:
    
    gr.Markdown("""
    # 📋 Whiteboard Meeting Notes AI
    
    Upload a photo of your whiteboard and let AI:
    - ✍️ **OCR handwritten text** using Microsoft's TrOCR model
    - 🔍 **Detect regions** automatically with YOLOv8
    - 🎯 **Extract action items** with assignees, deadlines, and priorities
    
    ### Powered by:
    - **TrOCR** (Handwriting Recognition)
    - **YOLOv8** (Region Detection)  
    - **Qwen2.5-7B** (Action Item Extraction)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            image_input = gr.Image(
                type="pil",
                label="📸 Upload Whiteboard Image",
                height=400
            )
            
            with gr.Accordion("⚙️ Advanced Options", open=True):
                use_regions = gr.Checkbox(
                    label="Use Region Detection",
                    value=True,
                    info="Automatically detect and segment different areas of the whiteboard"
                )
                extract_actions = gr.Checkbox(
                    label="Extract Action Items",
                    value=True,
                    info="Use AI to identify tasks, assignees, and deadlines"
                )
            
            analyze_btn = gr.Button(
                "🚀 Analyze Whiteboard",
                variant="primary",
                size="lg"
            )
            
            download_btn = gr.Button(
                "📥 Download Full Report",
                variant="secondary"
            )
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### 🎨 Annotated Whiteboard")
            annotated_image = gr.Image(
                label="Detected Regions",
                height=400
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✍️ Transcribed Text")
            ocr_output = gr.Textbox(
                label="OCR Results",
                lines=10,
                placeholder="OCR text will appear here..."
            )
        
        with gr.Column():
            gr.Markdown("### 🎯 Action Items")
            action_items_output = gr.HTML(
                label="Extracted Action Items"
            )
    
    # Download output
    report_file = gr.File(
        label="📄 Meeting Report",
        visible=False
    )
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_whiteboard_image,
        inputs=[image_input, use_regions, extract_actions],
        outputs=[annotated_image, ocr_output, action_items_output]
    )
    
    download_btn.click(
        fn=download_report,
        inputs=[image_input, use_regions, extract_actions],
        outputs=report_file
    ).then(
        lambda: gr.update(visible=True),
        outputs=report_file
    )
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### 💡 Tips for Best Results:")
    gr.Markdown("""
    - 📸 Take photos in good lighting
    - 🎯 Ensure text is clearly visible
    - 📐 Capture the whiteboard straight-on (avoid extreme angles)
    - ✏️ Write clearly (printed handwriting works best)
    - 🔲 For action items, use keywords like "TODO", "Action:", "@person", "by Friday"
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
