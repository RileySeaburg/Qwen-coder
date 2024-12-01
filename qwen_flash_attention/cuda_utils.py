import ctypes
import os
from typing import Optional, Tuple, Any, cast, Union, Dict
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Type definitions
InputType = Union[torch.Tensor, Dict[str, torch.Tensor]]
TensorType = Union[torch.Tensor, Dict[str, torch.Tensor]]

# Load CUDA memory manager library
CUDA_LIB_PATH = Path(__file__).parent / "cuda_kernels"
if os.name == 'nt':
    CUDA_LIB = str(CUDA_LIB_PATH / "memory_manager.dll")
else:
    CUDA_LIB = str(CUDA_LIB_PATH / "libmemory_manager.so")

logger.info(f"Looking for CUDA library at: {CUDA_LIB}")
if not os.path.exists(CUDA_LIB):
    logger.error(f"CUDA library not found at: {CUDA_LIB}")
    available_files = list(CUDA_LIB_PATH.glob('*'))
    logger.info(f"Available files in directory: {available_files}")

# Forward declaration for self-referential structure
class MemoryBlock(ctypes.Structure): pass
MemoryBlock._fields_ = [
    ("ptr", ctypes.c_void_p),
    ("size", ctypes.c_size_t),
    ("used", ctypes.c_bool),
    ("is_shared", ctypes.c_bool),
    ("next", ctypes.POINTER(MemoryBlock))
]

# Memory pool structure
class MemoryPool(ctypes.Structure):
    _fields_ = [
        ("dedicated_base", ctypes.c_void_p),
        ("shared_base", ctypes.c_void_p),
        ("dedicated_size", ctypes.c_size_t),
        ("shared_size", ctypes.c_size_t),
        ("blocks", ctypes.POINTER(MemoryBlock)),
        ("stream", ctypes.c_void_p)
    ]

class CUDAManager:
    """Manages CUDA memory and operations."""
    
    def __init__(self, device_id: int = 0, use_shared_memory: bool = True):
        """Initialize CUDA manager.
        
        Args:
            device_id: CUDA device ID
            use_shared_memory: Whether to use shared memory for large allocations
        """
        self.device_id = device_id
        self.use_shared_memory = use_shared_memory
        self.lib: Optional[ctypes.CDLL] = None
        self.pool: Optional[ctypes.pointer[MemoryPool]] = None
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid device ID {device_id}. Available devices: {torch.cuda.device_count()}")
            
        # Set device
        torch.cuda.set_device(device_id)
        
        # Initialize
        self._initialize()
        
    def _initialize(self):
        """Initialize CUDA memory manager."""
        try:
            # Load library with full error checking
            if not os.path.exists(CUDA_LIB):
                raise RuntimeError(f"CUDA library not found at: {CUDA_LIB}")
                
            try:
                logger.debug(f"Loading CUDA library from: {CUDA_LIB}")
                self.lib = ctypes.CDLL(CUDA_LIB)
                logger.debug(f"CUDA library loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CUDA library: {e}")
                raise
                
            # Check if functions exist
            required_functions = [
                'initMemoryPool',
                'allocateMemory',
                'freeMemory',
                'getMemoryInfo',
                'processData',
                'copyMemoryAsync',
                'synchronizeMemory',
                'cleanupMemoryPool'
            ]
            
            missing_functions = []
            available_functions = []
            for func in required_functions:
                if hasattr(self.lib, func):
                    available_functions.append(func)
                else:
                    missing_functions.append(func)
                    
            logger.debug(f"Available functions: {available_functions}")
            if missing_functions:
                logger.error(f"Missing required functions: {missing_functions}")
                raise RuntimeError(f"Missing required functions in CUDA library: {missing_functions}")
            
            # Set function prototypes
            logger.debug("Setting function prototypes")
            self.lib.initMemoryPool.argtypes = [ctypes.POINTER(MemoryPool), ctypes.c_size_t]
            self.lib.initMemoryPool.restype = ctypes.c_int
            
            self.lib.allocateMemory.argtypes = [ctypes.POINTER(MemoryPool), ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self.lib.allocateMemory.restype = ctypes.c_int
            
            self.lib.freeMemory.argtypes = [ctypes.POINTER(MemoryPool), ctypes.c_void_p]
            self.lib.freeMemory.restype = ctypes.c_int
            
            self.lib.getMemoryInfo.argtypes = [ctypes.POINTER(MemoryPool), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            self.lib.getMemoryInfo.restype = ctypes.c_int
            
            self.lib.processData.argtypes = [ctypes.POINTER(MemoryPool), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self.lib.processData.restype = ctypes.c_int
            
            self.lib.copyMemoryAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(MemoryPool)]
            self.lib.copyMemoryAsync.restype = ctypes.c_int
            
            self.lib.synchronizeMemory.argtypes = [ctypes.POINTER(MemoryPool)]
            self.lib.synchronizeMemory.restype = ctypes.c_int
            
            self.lib.cleanupMemoryPool.argtypes = [ctypes.POINTER(MemoryPool)]
            self.lib.cleanupMemoryPool.restype = ctypes.c_int
            
            # Initialize memory pool
            logger.debug("Initializing memory pool")
            pool = MemoryPool()
            self.pool = ctypes.pointer(pool)
            pool_size = 24 * 1024 * 1024 * 1024  # 24GB total (8GB GPU + 16GB shared)
            err = self.lib.initMemoryPool(self.pool, pool_size)
            if err != 0:
                raise RuntimeError(f"Failed to initialize memory pool: {err}")
                
            logger.info("CUDA memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDA memory manager: {e}")
            self.lib = None
            self.pool = None
            raise
            
    def allocate(self, size: int) -> Tuple[int, bool]:
        """Allocate memory.
        
        Args:
            size: Size in bytes
            
        Returns:
            Tuple of (pointer, is_shared)
        """
        if not self.lib or not self.pool:
            raise RuntimeError("CUDA memory manager not initialized")
            
        ptr = ctypes.c_void_p()
        err = self.lib.allocateMemory(self.pool, ctypes.byref(ptr), size)
        if err != 0:
            raise RuntimeError(f"Failed to allocate memory: {err}")
        
        # Find block to determine if it's shared
        block = self.pool.contents.blocks
        ptr_value = int(ptr.value) if ptr.value is not None else 0
        while block:
            block_ptr = int(block.contents.ptr.value) if block.contents.ptr.value is not None else 0
            if block_ptr == ptr_value:
                return ptr_value, bool(block.contents.is_shared)
            block = block.contents.next
            
        raise RuntimeError("Failed to find allocated block")
        
    def free(self, ptr: int):
        """Free memory."""
        if not self.lib or not self.pool:
            raise RuntimeError("CUDA memory manager not initialized")
            
        err = self.lib.freeMemory(self.pool, ctypes.c_void_p(ptr))
        if err != 0:
            raise RuntimeError(f"Failed to free memory: {err}")
            
    def get_memory_info(self) -> Tuple[int, int]:
        """Get memory info.
        
        Returns:
            Tuple of (free_size, largest_block)
        """
        if not self.lib or not self.pool:
            raise RuntimeError("CUDA memory manager not initialized")
            
        free_size = ctypes.c_size_t()
        largest_block = ctypes.c_size_t()
        err = self.lib.getMemoryInfo(self.pool, ctypes.byref(free_size), ctypes.byref(largest_block))
        if err != 0:
            raise RuntimeError(f"Failed to get memory info: {err}")
        return int(free_size.value), int(largest_block.value)
        
    def process_data(self, input_ptr: int, output_ptr: int, size: int):
        """Process data using shared memory."""
        if not self.lib or not self.pool:
            raise RuntimeError("CUDA memory manager not initialized")
            
        err = self.lib.processData(self.pool, ctypes.c_void_p(input_ptr), ctypes.c_void_p(output_ptr), size)
        if err != 0:
            raise RuntimeError(f"Failed to process data: {err}")
            
    def copy_async(self, dst: int, src: int, size: int):
        """Asynchronous memory copy."""
        if not self.lib or not self.pool:
            raise RuntimeError("CUDA memory manager not initialized")
            
        err = self.lib.copyMemoryAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), size, self.pool)
        if err != 0:
            raise RuntimeError(f"Failed to copy memory: {err}")
            
    def synchronize(self):
        """Synchronize memory operations."""
        if not self.lib or not self.pool:
            raise RuntimeError("CUDA memory manager not initialized")
            
        err = self.lib.synchronizeMemory(self.pool)
        if err != 0:
            raise RuntimeError(f"Failed to synchronize memory: {err}")
            
    def __del__(self):
        """Cleanup memory pool."""
        try:
            if hasattr(self, 'lib') and hasattr(self, 'pool') and self.lib and self.pool:
                self.lib.cleanupMemoryPool(self.pool)
        except:
            pass
