#include <cuda_runtime.h>
#include "memory_manager.cuh"
#include <stdio.h>

// Memory pool configuration
#define MAX_BLOCKS 1024
#define BLOCK_SIZE (1 << 20)  // 1MB blocks
#define MAX_ALLOCATIONS 4096

// Memory pool state
static void* device_memory_pool = nullptr;
static size_t* block_sizes = nullptr;
static bool* block_used = nullptr;
static size_t total_memory = 0;
static bool initialized = false;

// Error checking helper
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
}

bool initializeMemoryManager(size_t size) {
    if (initialized) {
        return true;
    }

    // Allocate device memory pool
    CHECK_CUDA(cudaMalloc(&device_memory_pool, size));
    
    // Allocate host tracking arrays
    block_sizes = new size_t[MAX_BLOCKS];
    block_used = new bool[MAX_BLOCKS];
    
    // Initialize tracking arrays
    for (int i = 0; i < MAX_BLOCKS; i++) {
        block_sizes[i] = 0;
        block_used[i] = false;
    }
    
    total_memory = size;
    initialized = true;
    
    // Zero out the memory pool
    CHECK_CUDA(cudaMemset(device_memory_pool, 0, size));
    
    return true;
}

void* allocateMemory(size_t size) {
    if (!initialized || size == 0) {
        return nullptr;
    }
    
    // Align size to 256-byte boundary
    size = (size + 255) & ~255;
    
    // Find best fit block
    int best_block = -1;
    size_t best_size = total_memory + 1;
    size_t current_offset = 0;
    
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (!block_used[i]) {
            size_t available = 0;
            if (i == 0) {
                available = total_memory;
            } else {
                available = total_memory - current_offset;
            }
            
            if (available >= size && available < best_size) {
                best_block = i;
                best_size = available;
            }
        }
        current_offset += block_sizes[i];
    }
    
    if (best_block == -1) {
        return nullptr;  // No suitable block found
    }
    
    // Calculate offset for new allocation
    size_t offset = 0;
    for (int i = 0; i < best_block; i++) {
        offset += block_sizes[i];
    }
    
    // Update block tracking
    block_sizes[best_block] = size;
    block_used[best_block] = true;
    
    return static_cast<char*>(device_memory_pool) + offset;
}

bool freeMemory(void* ptr) {
    if (!initialized || !ptr) {
        return false;
    }
    
    // Find block containing this pointer
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(device_memory_pool);
    size_t current_offset = 0;
    
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (current_offset == offset && block_used[i]) {
            block_used[i] = false;
            
            // Coalesce with adjacent free blocks
            int start = i;
            while (start > 0 && !block_used[start-1]) {
                start--;
            }
            
            int end = i;
            while (end < MAX_BLOCKS-1 && !block_used[end+1]) {
                end++;
            }
            
            // Merge blocks
            size_t merged_size = 0;
            for (int j = start; j <= end; j++) {
                merged_size += block_sizes[j];
                if (j != start) {
                    block_sizes[j] = 0;
                }
            }
            block_sizes[start] = merged_size;
            
            return true;
        }
        current_offset += block_sizes[i];
    }
    
    return false;
}

bool shutdownMemoryManager() {
    if (!initialized) {
        return true;
    }
    
    // Free device memory pool
    if (device_memory_pool) {
        CHECK_CUDA(cudaFree(device_memory_pool));
        device_memory_pool = nullptr;
    }
    
    // Free tracking arrays
    delete[] block_sizes;
    delete[] block_used;
    
    block_sizes = nullptr;
    block_used = nullptr;
    total_memory = 0;
    initialized = false;
    
    return true;
}

size_t getAvailableMemory() {
    if (!initialized) {
        return 0;
    }
    
    size_t used_memory = 0;
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (block_used[i]) {
            used_memory += block_sizes[i];
        }
    }
    
    return total_memory - used_memory;
}

bool defragmentMemory() {
    if (!initialized) {
        return false;
    }
    
    // Collect all used blocks
    struct Block {
        void* ptr;
        size_t size;
    };
    
    Block* used_blocks = new Block[MAX_BLOCKS];
    int used_count = 0;
    size_t current_offset = 0;
    
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (block_used[i]) {
            used_blocks[used_count].ptr = static_cast<char*>(device_memory_pool) + current_offset;
            used_blocks[used_count].size = block_sizes[i];
            used_count++;
        }
        current_offset += block_sizes[i];
    }
    
    // Allocate temporary buffer
    void* temp_buffer;
    CHECK_CUDA(cudaMalloc(&temp_buffer, total_memory));
    
    // Copy all used blocks to temporary buffer
    current_offset = 0;
    for (int i = 0; i < used_count; i++) {
        CHECK_CUDA(cudaMemcpy(
            static_cast<char*>(temp_buffer) + current_offset,
            used_blocks[i].ptr,
            used_blocks[i].size,
            cudaMemcpyDeviceToDevice
        ));
        current_offset += used_blocks[i].size;
    }
    
    // Reset block tracking
    for (int i = 0; i < MAX_BLOCKS; i++) {
        block_sizes[i] = 0;
        block_used[i] = false;
    }
    
    // Copy back in contiguous order
    CHECK_CUDA(cudaMemcpy(
        device_memory_pool,
        temp_buffer,
        current_offset,
        cudaMemcpyDeviceToDevice
    ));
    
    // Update block tracking
    current_offset = 0;
    for (int i = 0; i < used_count; i++) {
        block_sizes[i] = used_blocks[i].size;
        block_used[i] = true;
        current_offset += used_blocks[i].size;
    }
    
    // Free temporary buffer
    CHECK_CUDA(cudaFree(temp_buffer));
    delete[] used_blocks;
    
    return true;
}
