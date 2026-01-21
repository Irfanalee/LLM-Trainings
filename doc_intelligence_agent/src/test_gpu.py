"""
GPU Verification Script
Tests that PyTorch can properly access your RTX A4000
"""

import sys

def test_gpu():
    print("=" * 50)
    print("GPU Verification Test")
    print("=" * 50)
    
    # Test 1: PyTorch import
    print("\n[1/4] Testing PyTorch import...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import PyTorch: {e}")
        sys.exit(1)
    
    # Test 2: CUDA availability
    print("\n[2/4] Testing CUDA availability...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("  ✗ CUDA is NOT available")
        print("    This could mean:")
        print("    - NVIDIA drivers are not installed")
        print("    - PyTorch was installed without CUDA support")
        print("    - CUDA toolkit version mismatch")
        sys.exit(1)
    
    # Test 3: GPU detection
    print("\n[3/4] Testing GPU detection...")
    gpu_count = torch.cuda.device_count()
    print(f"  ✓ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    - Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    - Compute Capability: {props.major}.{props.minor}")
        print(f"    - Multi-Processor Count: {props.multi_processor_count}")
    
    # Test 4: Tensor operations
    print("\n[4/4] Testing tensor operations on GPU...")
    try:
        # Create tensors on GPU
        device = torch.device("cuda:0")
        
        # Small test
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        
        print(f"  ✓ Small tensor multiplication: {a.shape} x {b.shape}")
        
        # Larger test (simulating model operations)
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        c = torch.matmul(a, b)
        
        print(f"  ✓ Large tensor multiplication: {a.shape} x {b.shape}")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\n  Memory after test:")
        print(f"    - Allocated: {allocated:.2f} GB")
        print(f"    - Reserved:  {reserved:.2f} GB")
        
        # Cleanup
        del a, b, c
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Tensor operation failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All tests passed! Your GPU is ready.")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    test_gpu()
