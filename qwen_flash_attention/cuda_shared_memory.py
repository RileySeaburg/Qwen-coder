import torch
import logging
from typing import Optional, Dict, Any, Tuple
from .cuda_utils import CUDAManager

logger = logging.getLogger(__name__)

class SharedMemoryManager:
    """Manages shared memory allocation for large tensors."""
    
    def __init__(self, device_id: int = 0):
        """Initialize shared memory manager.
        
        Args:
            device_id: CUDA device ID
        """
        self.device_id = device_id
        self.cuda_manager = CUDAManager(device_id)
        self.allocated_tensors: Dict[int, Tuple[torch.Tensor, bool]] = {}  # ptr -> (tensor, is_shared)
        
    def allocate(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a tensor in shared memory.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor backed by shared memory
        """
        # Calculate size in bytes
        size = torch.tensor([], dtype=dtype).element_size()
        for dim in shape:
            size *= dim
            
        # Allocate memory
        ptr, is_shared = self.cuda_manager.allocate(size)
        
        # Create tensor from memory
        if is_shared:
            # Create tensor using shared memory
            tensor = torch.zeros(shape, dtype=dtype, device=f'cuda:{self.device_id}')
            tensor.storage().resize_(size // tensor.element_size())
            tensor.data = torch.cuda.ShortTensor(shape).set_(
                torch.cuda.ShortStorage.from_address(ptr, size // tensor.element_size())
            )
        else:
            # Create tensor using dedicated GPU memory
            tensor = torch.zeros(shape, dtype=dtype, device=f'cuda:{self.device_id}')
            tensor.storage().resize_(size // tensor.element_size())
            tensor.data = torch.cuda.ShortTensor(shape).set_(
                torch.cuda.ShortStorage.from_address(ptr, size // tensor.element_size())
            )
        
        # Track allocation
        self.allocated_tensors[ptr] = (tensor, is_shared)
        
        logger.info(f"Allocated tensor of shape {shape} ({size / 1024**2:.1f}MB) in {'shared' if is_shared else 'dedicated'} memory")
        return tensor
        
    def free(self, tensor: torch.Tensor):
        """Free a tensor's memory.
        
        Args:
            tensor: Tensor to free
        """
        # Find tensor's pointer
        ptr = None
        for p, (t, _) in self.allocated_tensors.items():
            if t.data_ptr() == tensor.data_ptr():
                ptr = p
                break
                
        if ptr is None:
            raise RuntimeError("Tensor not found in allocated tensors")
            
        # Free memory
        self.cuda_manager.free(ptr)
        del self.allocated_tensors[ptr]
        
    def process_tensor(self, tensor: torch.Tensor, fn: str = "double") -> torch.Tensor:
        """Process a tensor using shared memory.
        
        Args:
            tensor: Input tensor
            fn: Processing function ("double" multiplies by 2)
            
        Returns:
            Processed tensor
        """
        # Find input tensor's pointer
        in_ptr = None
        for p, (t, _) in self.allocated_tensors.items():
            if t.data_ptr() == tensor.data_ptr():
                in_ptr = p
                break
                
        if in_ptr is None:
            raise RuntimeError("Input tensor not found in allocated tensors")
            
        # Allocate output tensor
        out_tensor = self.allocate(tensor.shape, tensor.dtype)
        out_ptr = None
        for p, (t, _) in self.allocated_tensors.items():
            if t.data_ptr() == out_tensor.data_ptr():
                out_ptr = p
                break
                
        if out_ptr is None:
            raise RuntimeError("Output tensor not found in allocated tensors")
            
        # Process data
        self.cuda_manager.process_data(in_ptr, out_ptr, tensor.numel() * tensor.element_size())
        
        return out_tensor
        
    def copy_tensor(self, src: torch.Tensor, dst: torch.Tensor):
        """Copy data between tensors.
        
        Args:
            src: Source tensor
            dst: Destination tensor
        """
        # Find tensor pointers
        src_ptr = None
        dst_ptr = None
        for p, (t, _) in self.allocated_tensors.items():
            if t.data_ptr() == src.data_ptr():
                src_ptr = p
            if t.data_ptr() == dst.data_ptr():
                dst_ptr = p
                
        if src_ptr is None:
            raise RuntimeError("Source tensor not found in allocated tensors")
        if dst_ptr is None:
            raise RuntimeError("Destination tensor not found in allocated tensors")
            
        # Copy data
        self.cuda_manager.copy_async(dst_ptr, src_ptr, src.numel() * src.element_size())
        
    def synchronize(self):
        """Synchronize memory operations."""
        self.cuda_manager.synchronize()
        
    def get_memory_info(self) -> Tuple[int, int]:
        """Get memory usage information.
        
        Returns:
            Tuple of (free_size, largest_block) in bytes
        """
        return self.cuda_manager.get_memory_info()
        
    def __del__(self):
        """Cleanup allocated memory."""
        # Free all tensors
        for ptr, (tensor, _) in self.allocated_tensors.items():
            try:
                self.cuda_manager.free(ptr)
            except:
                pass
        self.allocated_tensors.clear()
