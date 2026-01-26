#!/bin/bash

echo "======================================"
echo "LLM-Trainings - Setup"
echo "======================================"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. CPU mode will be used."
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing dependencies..."
echo ""

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install main requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install whiteboard-ai requirements
if [ -f "whiteboard-ai/requirements.txt" ]; then
    echo ""
    echo "📦 Installing whiteboard-ai dependencies..."
    pip install -r whiteboard-ai/requirements.txt
fi

# Install doc_intelligence_agent requirements
if [ -f "doc_intelligence_agent/requirements.txt" ]; then
    echo ""
    echo "📦 Installing doc_intelligence_agent dependencies..."
    pip install -r doc_intelligence_agent/requirements.txt
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "======================================"
echo "Projects:"
echo "======================================"
echo ""
echo "1. Whiteboard AI:"
echo "   cd whiteboard-ai && python gradio_app.py"
echo "   Access: http://localhost:7860"
echo ""
echo "2. Doc Intelligence Agent:"
echo "   cd doc_intelligence_agent"
echo ""
echo "======================================"
echo "Training:"
echo "======================================"
echo ""
echo "Whiteboard AI Training:"
echo "   python whiteboard-ai/scripts/generate_synthetic_data.py"
echo "   python whiteboard-ai/training/train_yolo.py"
echo "   python whiteboard-ai/training/train_qwen_lora.py train"
echo ""
echo "======================================"
echo "Activate environment: source venv/bin/activate"
echo "======================================"
