# Whiteboard Meeting Notes AI - Project Structure

```
whiteboard-ai/
│
├── README.md                  # Main documentation
├── requirements.txt           # Python dependencies
├── setup.sh                   # Installation script
│
├── whiteboard_ai.py          # Core AI pipeline
│   ├── WhiteboardAI class
│   ├── Region detection (YOLOv8)
│   ├── OCR (TrOCR)
│   └── Action extraction (Qwen2.5-7B)
│
├── gradio_app.py             # Web interface
│   ├── Drag-and-drop UI
│   ├── Real-time visualization
│   └── Report downloads
│
├── examples.py               # Usage examples
│   ├── Basic analysis
│   ├── Custom pipeline
│   ├── Batch processing
│   └── Export formats
│
├── generate_sample.py        # Test data generator
│   └── Synthetic whiteboards
│
└── outputs/                  # Generated files
    ├── annotated_*.jpg       # Visualizations
    ├── meeting_report.txt    # Text reports
    ├── action_items.csv      # CSV exports
    └── whiteboard_analysis.json  # JSON exports
```

## File Descriptions

### Core Files

**whiteboard_ai.py**
- Main AI pipeline implementation
- Classes: `WhiteboardAI`, `Region`, `ActionItem`
- Methods:
  - `detect_regions()`: YOLOv8 object detection
  - `ocr_region()`: TrOCR handwriting recognition
  - `extract_action_items()`: LLM-based extraction
  - `analyze_whiteboard()`: Complete pipeline
  - `visualize_results()`: Generate annotated images
  - `generate_report()`: Create text reports

**gradio_app.py**
- Gradio web interface
- Features:
  - Image upload with preview
  - Toggle region detection
  - Toggle action extraction
  - Interactive results display
  - Download reports

**examples.py**
- Comprehensive usage examples
- 9 different use cases
- Integration patterns
- Export formats

### Setup Files

**requirements.txt**
```
torch>=2.0.0
transformers>=4.35.0
ultralytics>=8.0.0
opencv-python>=4.8.0
gradio>=4.0.0
bitsandbytes>=0.41.0
```

**setup.sh**
- Automated installation
- GPU detection
- Dependency management

### Utility Files

**generate_sample.py**
- Creates synthetic whiteboards
- Testing without real photos
- Multiple scenarios

## Quick Reference

### Start Web Interface
```bash
python gradio_app.py
```

### Python API
```python
from whiteboard_ai import WhiteboardAI

ai = WhiteboardAI()
results = ai.analyze_whiteboard("image.jpg")
```

### Generate Test Data
```python
python generate_sample.py
```

### Run Examples
```python
python examples.py
```

## Model Files (Auto-downloaded)

When you first run the code, these models will be downloaded:

1. **YOLOv8n.pt** (~6 MB)
   - Location: `~/.cache/ultralytics/`
   
2. **TrOCR Large Handwritten** (~1.3 GB)
   - Location: `~/.cache/huggingface/`
   
3. **Qwen2.5-7B-Instruct** (~4.5 GB quantized)
   - Location: `~/.cache/huggingface/`

Total: ~6 GB download on first run

## GPU Memory Usage

| Component | VRAM |
|-----------|------|
| YOLOv8 | 0.5 GB |
| TrOCR | 3 GB |
| Qwen2.5-7B (4-bit) | 6 GB |
| **Peak Total** | **~11 GB** |

Fits comfortably in RTX A4000 16GB!

## Development Workflow

1. **Generate test data**
   ```bash
   python generate_sample.py
   ```

2. **Test core pipeline**
   ```python
   python -c "from whiteboard_ai import WhiteboardAI; ai = WhiteboardAI()"
   ```

3. **Run web interface**
   ```bash
   python gradio_app.py
   ```

4. **Analyze results**
   - Check annotated images
   - Review text reports
   - Verify action items

## Customization Points

### Add Custom Preprocessing
Edit `preprocess_for_ocr()` in `whiteboard_ai.py`

### Modify Action Extraction Prompt
Edit the prompt in `extract_action_items()`

### Change Models
- OCR: Replace `microsoft/trocr-large-handwritten`
- LLM: Replace `Qwen/Qwen2.5-7B-Instruct`
- Detector: Replace `yolov8n.pt` with larger variants

### Extend Output Formats
Add export functions to `WhiteboardAI` class

## Testing Strategy

1. **Unit Tests**: Test each component
2. **Integration Tests**: Full pipeline
3. **Sample Data**: Use generated whiteboards
4. **Real Data**: Test with actual photos
5. **Edge Cases**: Poor lighting, angles, etc.

## Performance Tuning

### For Faster Inference:
- Use YOLOv8n (nano) - already default
- Reduce max_new_tokens in LLM
- Skip region detection for simple boards

### For Better Accuracy:
- Use YOLOv8x (extra large)
- Increase OCR resolution
- Use larger LLM (Qwen2.5-14B)

### For Lower Memory:
- Use 8-bit quantization
- Process regions sequentially
- Use CPU for some components

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size, use CPU |
| Poor OCR | Better lighting, higher res |
| No actions found | Use clearer keywords |
| Slow inference | Use smaller models |
| Model download fails | Check internet, clear cache |

## Next Steps

After setup:
1. ✅ Run `python generate_sample.py`
2. ✅ Test with `python examples.py`
3. ✅ Launch UI with `python gradio_app.py`
4. ✅ Try your own whiteboard photos!

Happy analyzing! 📋✨
