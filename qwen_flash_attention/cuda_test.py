import torch
import sys
from .cuda_utils import CUDAManager

def test_cuda_availability():
    """Test basic CUDA availability"""
    print("Testing CUDA Availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Properties: {torch.cuda.get_device_properties(i)}")
    return torch.cuda.is_available()

def test_cuda_memory():
    """Test CUDA memory operations"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return False
    
    print("\nTesting CUDA Memory Operations...")
    try:
        # Try to allocate a small tensor
        x = torch.ones(1000, 1000).cuda()
        y = x + x
        del x, y
        torch.cuda.empty_cache()
        print("✓ Basic CUDA memory operations successful")
        return True
    except Exception as e:
        print(f"✗ CUDA memory test failed: {str(e)}")
        return False

def test_cuda_compute():
    """Test CUDA computation"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping compute test")
        return False
    
    print("\nTesting CUDA Computation...")
    try:
        # Matrix multiplication test
        size = 1000
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        
        # Create CUDA stream
        stream = torch.cuda.current_stream()
        
        # Create events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start event
        start_event.record(stream=stream)
        
        # Perform computation
        c = torch.mm(a, b)
        
        # Record end event
        end_event.record(stream=stream)
        
        # Wait for computation to finish
        torch.cuda.synchronize()
        
        # Calculate elapsed time
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"✓ Matrix multiplication ({size}x{size}) completed in {elapsed_time:.2f}ms")
        
        del a, b, c
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ CUDA compute test failed: {str(e)}")
        return False

def test_cuda_manager():
    """Test CUDAManager functionality"""
    print("\nTesting CUDAManager...")
    try:
        manager = CUDAManager(0)
        print("✓ CUDAManager initialized successfully")
        
        # Test memory management
        allocated, reserved = manager.get_memory_usage()
        print(f"Memory usage - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        # Test tensor operations
        test_tensor = torch.ones(100, 100)
        gpu_tensor = manager.prepare_input(test_tensor)
        print(f"✓ Tensor successfully moved to GPU: {gpu_tensor.is_cuda}")
        
        # Test memory optimization
        manager.optimize_memory()
        print("✓ Memory optimization successful")
        
        return True
    except Exception as e:
        print(f"✗ CUDAManager test failed: {str(e)}")
        return False

def main():
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("CUDA Memory", test_cuda_memory),
        ("CUDA Compute", test_cuda_compute),
        ("CUDA Manager", test_cuda_manager)
    ]
    
    results = []
    for name, test in tests:
        print(f"\n{'='*20} {name} Test {'='*20}")
        try:
            result = test()
            results.append((name, result))
        except Exception as e:
            print(f"Test failed with error: {str(e)}")
            results.append((name, False))
    
    print("\n{'='*20} Test Summary {'='*20}")
    all_passed = True
    for name, result in results:
        status = "✓ Passed" if result else "✗ Failed"
        print(f"{name}: {status}")
        all_passed = all_passed and result
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
