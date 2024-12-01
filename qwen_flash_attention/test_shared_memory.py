import torch
import logging
from cuda_shared_memory import SharedMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_allocation():
    """Test basic memory allocation and freeing."""
    manager = SharedMemoryManager()
    
    # Test small tensor (should use dedicated memory)
    shape = (1000, 1000)  # ~4MB
    dtype = torch.float32
    logger.info(f"\nTesting small tensor allocation {shape}")
    tensor = manager.allocate(shape, dtype)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    free_size, largest_block = manager.get_memory_info()
    logger.info(f"Free memory: {free_size / 1024**3:.1f}GB (largest block: {largest_block / 1024**3:.1f}GB)")
    manager.free(tensor)
    
    # Test medium tensor (should use dedicated memory)
    shape = (5000, 5000)  # ~100MB
    logger.info(f"\nTesting medium tensor allocation {shape}")
    tensor = manager.allocate(shape, dtype)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    free_size, largest_block = manager.get_memory_info()
    logger.info(f"Free memory: {free_size / 1024**3:.1f}GB (largest block: {largest_block / 1024**3:.1f}GB)")
    manager.free(tensor)
    
    # Test large tensor (should use shared memory)
    shape = (50000, 50000)  # ~10GB
    logger.info(f"\nTesting large tensor allocation {shape}")
    tensor = manager.allocate(shape, dtype)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    free_size, largest_block = manager.get_memory_info()
    logger.info(f"Free memory: {free_size / 1024**3:.1f}GB (largest block: {largest_block / 1024**3:.1f}GB)")
    manager.free(tensor)

def test_tensor_operations():
    """Test tensor operations using shared memory."""
    manager = SharedMemoryManager()
    
    # Create test tensors
    shapes = [
        (1000, 1000),    # Small (~4MB)
        (5000, 5000),    # Medium (~100MB)
        (10000, 10000),  # Large (~400MB)
    ]
    
    for shape in shapes:
        logger.info(f"\nTesting tensor operations for shape {shape}")
        
        # Allocate input tensor
        input_tensor = manager.allocate(shape, torch.float32)
        input_tensor.fill_(1.0)  # Fill with ones
        
        # Process tensor (double values)
        output_tensor = manager.process_tensor(input_tensor)
        manager.synchronize()
        
        # Verify results
        expected = torch.full(shape, 2.0, dtype=torch.float32, device=f'cuda:{manager.device_id}')
        assert torch.allclose(output_tensor, expected)
        logger.info("Tensor processing test passed!")
        
        # Test copying
        dst_tensor = manager.allocate(shape, torch.float32)
        manager.copy_tensor(output_tensor, dst_tensor)
        manager.synchronize()
        assert torch.allclose(dst_tensor, expected)
        logger.info("Tensor copy test passed!")
        
        # Cleanup
        manager.free(input_tensor)
        manager.free(output_tensor)
        manager.free(dst_tensor)

def test_concurrent_tensors():
    """Test concurrent tensor allocations."""
    manager = SharedMemoryManager()
    
    # Allocate multiple tensors
    tensors = []
    shapes = [
        (5000, 5000),    # 100MB
        (10000, 10000),  # 400MB
        (15000, 15000),  # 900MB
        (20000, 20000),  # 1.6GB
    ]
    
    logger.info("\nTesting concurrent tensor allocations")
    for i, shape in enumerate(shapes):
        logger.info(f"Allocating tensor {i+1}: {shape}")
        tensor = manager.allocate(shape, torch.float32)
        tensors.append(tensor)
        free_size, largest_block = manager.get_memory_info()
        logger.info(f"Free memory: {free_size / 1024**3:.1f}GB (largest block: {largest_block / 1024**3:.1f}GB)")
    
    # Process all tensors
    logger.info("\nProcessing tensors concurrently")
    outputs = []
    for tensor in tensors:
        output = manager.process_tensor(tensor)
        outputs.append(output)
    manager.synchronize()
    
    # Verify results
    for tensor, output in zip(tensors, outputs):
        expected = tensor * 2
        assert torch.allclose(output, expected)
    logger.info("All concurrent operations passed!")
    
    # Cleanup
    for tensor in tensors + outputs:
        manager.free(tensor)

def main():
    """Run all tests."""
    try:
        test_memory_allocation()
        test_tensor_operations()
        test_concurrent_tensors()
        logger.info("\nAll tests passed!")
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        raise

if __name__ == "__main__":
    main()
