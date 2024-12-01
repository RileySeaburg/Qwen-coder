#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

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

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
        return err; \
    } \
}

// Initialize memory pool
EXPORT cudaError_t __cdecl initMemoryPool(MemoryPool* pool, size_t size) {
    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
    printf("L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Max shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Memory clock rate: %d KHz\n", prop.memoryClockRate);
    
    // Calculate available memory
    size_t free_gpu, total_gpu;
    CHECK_CUDA(cudaMemGetInfo(&free_gpu, &total_gpu));
    printf("Free GPU memory: %zu MB\n", free_gpu / (1024*1024));
    printf("Total GPU memory: %zu MB\n", total_gpu / (1024*1024));
    
    // Initialize dedicated GPU memory pool (75% of free GPU memory)
    size_t dedicated_size = free_gpu * 3 / 4;
    CHECK_CUDA(cudaMalloc(&pool->dedicated_base, dedicated_size));
    pool->dedicated_size = dedicated_size;
    printf("Allocated %zu MB dedicated GPU memory\n", dedicated_size / (1024*1024));
    
    // Initialize shared system memory pool (75% of available shared memory)
    size_t shared_size = 16ULL * 1024 * 1024 * 1024 * 3 / 4;  // 12GB of 16GB shared memory
    CHECK_CUDA(cudaMallocHost(&pool->shared_base, shared_size, cudaHostAllocMapped));
    pool->shared_size = shared_size;
    printf("Allocated %zu MB shared system memory\n", shared_size / (1024*1024));
    
    // Initialize first blocks
    MemoryBlock* dedicated_block = new MemoryBlock;
    dedicated_block->ptr = pool->dedicated_base;
    dedicated_block->size = dedicated_size;
    dedicated_block->used = false;
    dedicated_block->is_shared = false;
    dedicated_block->next = nullptr;
    
    MemoryBlock* shared_block = new MemoryBlock;
    shared_block->ptr = pool->shared_base;
    shared_block->size = shared_size;
    shared_block->used = false;
    shared_block->is_shared = true;
    shared_block->next = dedicated_block;
    
    pool->blocks = shared_block;
    
    // Create CUDA stream
    CHECK_CUDA(cudaStreamCreate(&pool->stream));
    
    return cudaSuccess;
}

// Allocate memory from pool
EXPORT cudaError_t __cdecl allocateMemory(MemoryPool* pool, void** ptr, size_t size) {
    // Align size to cache line
    size = (size + 511) & ~511;
    
    // Try dedicated GPU memory first for small allocations
    bool prefer_shared = (size > pool->dedicated_size / 4);  // Use shared memory for large allocations
    
    // First pass: try preferred memory type
    MemoryBlock* block = pool->blocks;
    while (block != nullptr) {
        if (!block->used && block->size >= size && block->is_shared == prefer_shared) {
            // Split block if needed
            if (block->size > size + sizeof(MemoryBlock) + 512) {
                MemoryBlock* new_block = new MemoryBlock;
                new_block->ptr = (void*)((char*)block->ptr + size);
                new_block->size = block->size - size;
                new_block->used = false;
                new_block->is_shared = block->is_shared;
                new_block->next = block->next;
                
                block->size = size;
                block->next = new_block;
            }
            
            block->used = true;
            *ptr = block->ptr;
            
            // Initialize memory
            CHECK_CUDA(cudaMemsetAsync(*ptr, 0, size, pool->stream));
            
            printf("Allocated %zu MB %s memory\n", 
                   size / (1024*1024), 
                   block->is_shared ? "shared" : "dedicated");
            
            return cudaSuccess;
        }
        block = block->next;
    }
    
    // Second pass: try other memory type
    block = pool->blocks;
    while (block != nullptr) {
        if (!block->used && block->size >= size) {
            // Split block if needed
            if (block->size > size + sizeof(MemoryBlock) + 512) {
                MemoryBlock* new_block = new MemoryBlock;
                new_block->ptr = (void*)((char*)block->ptr + size);
                new_block->size = block->size - size;
                new_block->used = false;
                new_block->is_shared = block->is_shared;
                new_block->next = block->next;
                
                block->size = size;
                block->next = new_block;
            }
            
            block->used = true;
            *ptr = block->ptr;
            
            // Initialize memory
            CHECK_CUDA(cudaMemsetAsync(*ptr, 0, size, pool->stream));
            
            printf("Allocated %zu MB %s memory (fallback)\n", 
                   size / (1024*1024), 
                   block->is_shared ? "shared" : "dedicated");
            
            return cudaSuccess;
        }
        block = block->next;
    }
    
    return cudaErrorMemoryAllocation;
}

// Free memory back to pool
EXPORT cudaError_t __cdecl freeMemory(MemoryPool* pool, void* ptr) {
    if (!ptr) return cudaSuccess;  // Ignore null pointers
    
    MemoryBlock* block = pool->blocks;
    MemoryBlock* prev = nullptr;
    
    while (block != nullptr) {
        if (block->ptr == ptr) {
            block->used = false;
            
            // Merge with next block if free and same type
            while (block->next != nullptr && 
                   !block->next->used && 
                   block->is_shared == block->next->is_shared) {
                MemoryBlock* next = block->next;
                block->size += next->size;
                block->next = next->next;
                delete next;
            }
            
            // Merge with previous block if free and same type
            if (prev != nullptr && 
                !prev->used && 
                prev->is_shared == block->is_shared) {
                prev->size += block->size;
                prev->next = block->next;
                delete block;
            }
            
            printf("Freed %zu MB %s memory\n", 
                   block->size / (1024*1024), 
                   block->is_shared ? "shared" : "dedicated");
            
            return cudaSuccess;
        }
        prev = block;
        block = block->next;
    }
    
    return cudaErrorInvalidValue;
}

// Get memory info
EXPORT cudaError_t __cdecl getMemoryInfo(MemoryPool* pool, size_t* free_size, size_t* largest_block) {
    size_t free_dedicated = 0;
    size_t free_shared = 0;
    size_t largest_dedicated = 0;
    size_t largest_shared = 0;
    
    MemoryBlock* block = pool->blocks;
    while (block != nullptr) {
        if (!block->used) {
            if (block->is_shared) {
                free_shared += block->size;
                largest_shared = max(largest_shared, block->size);
            } else {
                free_dedicated += block->size;
                largest_dedicated = max(largest_dedicated, block->size);
            }
        }
        block = block->next;
    }
    
    printf("Free dedicated: %zu MB (largest: %zu MB)\n", 
           free_dedicated / (1024*1024), 
           largest_dedicated / (1024*1024));
    printf("Free shared: %zu MB (largest: %zu MB)\n", 
           free_shared / (1024*1024), 
           largest_shared / (1024*1024));
    
    *free_size = free_dedicated + free_shared;
    *largest_block = max(largest_dedicated, largest_shared);
    
    return cudaSuccess;
}

// Process data using shared memory
EXPORT cudaError_t __cdecl processData(MemoryPool* pool, void* input_ptr, void* output_ptr, size_t size) {
    // Process data in chunks to handle large sizes
    const size_t chunk_size = 48 * 1024;  // 48KB chunks (max shared memory)
    const int threads = 256;
    
    for (size_t offset = 0; offset < size; offset += chunk_size) {
        size_t current_chunk = min(chunk_size, size - offset);
        int blocks = (current_chunk + threads - 1) / threads;
        
        // Launch kernel for this chunk
        void* in_chunk = (void*)((char*)input_ptr + offset);
        void* out_chunk = (void*)((char*)output_ptr + offset);
        
        // Simple multiplication by 2 for testing
        float* in_data = (float*)in_chunk;
        float* out_data = (float*)out_chunk;
        for (int i = 0; i < current_chunk/sizeof(float); i++) {
            out_data[i] = in_data[i] * 2;
        }
    }
    
    return cudaSuccess;
}

// Memory copy with stream
EXPORT cudaError_t __cdecl copyMemoryAsync(void* dst, const void* src, size_t size, MemoryPool* pool) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, pool->stream);
}

// Synchronize memory operations
EXPORT cudaError_t __cdecl synchronizeMemory(MemoryPool* pool) {
    return cudaStreamSynchronize(pool->stream);
}

// Cleanup memory pool
EXPORT cudaError_t __cdecl cleanupMemoryPool(MemoryPool* pool) {
    if (!pool) return cudaSuccess;
    
    // Free all blocks
    while (pool->blocks != nullptr) {
        MemoryBlock* next = pool->blocks->next;
        delete pool->blocks;
        pool->blocks = next;
    }
    
    // Free dedicated GPU memory
    if (pool->dedicated_base) {
        CHECK_CUDA(cudaFree(pool->dedicated_base));
        pool->dedicated_base = nullptr;
    }
    
    // Free shared system memory
    if (pool->shared_base) {
        CHECK_CUDA(cudaFreeHost(pool->shared_base));
        pool->shared_base = nullptr;
    }
    
    // Destroy stream
    if (pool->stream) {
        CHECK_CUDA(cudaStreamDestroy(pool->stream));
        pool->stream = nullptr;
    }
    
    return cudaSuccess;
}
