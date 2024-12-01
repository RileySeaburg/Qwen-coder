#ifndef MEMORY_MANAGER_CUH
#define MEMORY_MANAGER_CUH

#include <cuda_runtime.h>

// Memory block structure
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool used;
    bool is_shared;  // Whether this block uses shared system memory
    MemoryBlock* next;
};

// Memory pool structure
struct MemoryPool {
    void* dedicated_base;  // Base pointer for dedicated GPU memory
    void* shared_base;     // Base pointer for shared system memory
    size_t dedicated_size; // Size of dedicated GPU memory pool
    size_t shared_size;    // Size of shared system memory pool
    MemoryBlock* blocks;
    cudaStream_t stream;
};

// CUDA kernel declarations (defined in .cu file)
template<typename T>
__global__ void processWithSharedMemory(T* input, T* output, size_t size);

#ifdef __cplusplus
extern "C" {
#endif

// Initialize memory pool
cudaError_t initMemoryPool(MemoryPool* pool, size_t size);

// Allocate memory from pool
cudaError_t allocateMemory(MemoryPool* pool, void** ptr, size_t size);

// Free memory back to pool
cudaError_t freeMemory(MemoryPool* pool, void* ptr);

// Get memory info
cudaError_t getMemoryInfo(MemoryPool* pool, size_t* free_size, size_t* largest_block);

// Process data using shared memory
cudaError_t processData(MemoryPool* pool, void* input_ptr, void* output_ptr, size_t size);

// Memory copy with stream
cudaError_t copyMemoryAsync(void* dst, const void* src, size_t size, MemoryPool* pool);

// Synchronize memory operations
cudaError_t synchronizeMemory(MemoryPool* pool);

// Cleanup memory pool
cudaError_t cleanupMemoryPool(MemoryPool* pool);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_MANAGER_CUH
