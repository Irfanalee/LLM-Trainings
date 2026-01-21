#!/bin/bash
# Document Intelligence Agent - Environment Setup
# For Ubuntu with RTX A4000 16GB

set -e

echo "=========================================="
echo "Document Intelligence Agent Setup"
echo "=========================================="

# Check NVIDIA driver and CUDA
echo ""
echo "[1/6] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# Check Python version
echo ""
echo "[2/6] Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (CUDA 12.1 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
pip install transformers accelerate bitsandbytes
pip install datasets evaluate
pip install peft  # For LoRA fine-tuning
pip install sentencepiece protobuf

# Install document processing libraries
pip install pdf2image pymupdf pillow
pip install python-docx openpyxl  # For Office docs

# Install vision-language model dependencies
pip install einops timm

# Install inference/serving
pip install gradio fastapi uvicorn

# Install utilities
pip install tqdm rich python-dotenv

echo ""
echo "[5/6] Verifying PyTorch CUDA access..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "[6/6] Creating project structure..."
mkdir -p data/{raw,processed,train,val}
mkdir -p models/{checkpoints,final}
mkdir -p src/{data,models,training,inference}
mkdir -p notebooks
mkdir -p demos

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  cd doc_intelligence_agent"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Run: python src/test_gpu.py"
echo "  2. Run: python src/download_models.py"
echo ""
