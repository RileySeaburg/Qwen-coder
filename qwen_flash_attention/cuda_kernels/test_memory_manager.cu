#include <cuda_runtime.h>
#include <stdio.h>
#include "memory_manager.cuh"

// Test kernel to write to memory
__global__ void writeTest(int* data, int size, int value) {
    extern __shared__ __align__(sizeof(int)) unsigned char shared_mem[];
    int* shared = reinterpret_cast<int*>(shared_mem);
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x;
    
    // Initialize shared memory
    if (tid < blockDim.x) {
        shared[tid] = value;
    }
    __syncthreads();
    
    // Write to global memory using shared memory
    if (tid < blockDim.x && (offset + tid) < size) {
        data[offset + tid] = shared[tid % blockDim.x];
    }
}

// Test kernel to verify memory
__global__ void verifyTest(int* data, int size, int value, int* result) {
    extern __shared__ __align__(sizeof(int)) unsigned char shared_mem[];
    int* shared = reinterpret_cast<int*>(shared_mem);
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x;
    
    // Load data into shared memory
    if (tid < blockDim.x && (offset + tid) < size) {
        shared[tid] = data[offset + tid];
    }
    __syncthreads();
    
    // Verify data in shared memory
    if (tid < blockDim.x && (offset + tid) < size) {
        if (shared[tid] != value) {
            atomicAdd(result, 1);  // Count mismatches
        }
    }
}

// Test memory allocation and processing
void testMemoryAllocation(MemoryPool* pool, size_t size, int test_num) {
    printf("\nTest %d: Allocating %zu GB\n", test_num, size/(1024*1024*1024));
    
    // Allocate input and output memory
    void *input_ptr, *output_ptr;
    cudaError_t err = allocateMemory(pool, &input_ptr, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate input memory\n");
        return;
    }
    err = allocateMemory(pool, &output_ptr, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate output memory\n");
        freeMemory(pool, input_ptr);
        return;
    }
    
    // Cast to int pointer for test
    int* input = (int*)input_ptr;
    int* output = (int*)output_ptr;
    int num_ints = size / sizeof(int);
    
    // Launch write kernel with shared memory
    int threads = 256;
    int blocks = (num_ints + threads - 1) / threads;
    int shared_mem_size = threads * sizeof(int);
    
    writeTest<<<blocks, threads, shared_mem_size>>>(input, num_ints, test_num);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Write kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    cudaDeviceSynchronize();
    
    // Process data using shared memory
    err = processData(pool, input_ptr, output_ptr, size);
    if (err != cudaSuccess) {
        printf("Failed to process data: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    cudaDeviceSynchronize();
    
    // Verify output
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));
    
    verifyTest<<<blocks, threads, shared_mem_size>>>(output, num_ints, test_num*2, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Verify kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        goto cleanup;
    }
    cudaDeviceSynchronize();
    
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    if (h_result == 0) {
        printf("Memory test passed!\n");
    } else {
        printf("Memory test failed: %d mismatches\n", h_result);
    }
    
    // Get memory info
    size_t free_size, largest_block;
    getMemoryInfo(pool, &free_size, &largest_block);
    
cleanup:
    // Free memory
    freeMemory(pool, input_ptr);
    freeMemory(pool, output_ptr);
}

int main() {
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Create memory pool
    MemoryPool pool;
    size_t pool_size = 24ULL * 1024 * 1024 * 1024;  // 24GB total (8GB GPU + 16GB shared)
    cudaError_t err = initMemoryPool(&pool, pool_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize memory pool\n");
        return 1;
    }
    
    // Test small allocations (should use dedicated GPU memory)
    testMemoryAllocation(&pool, 1ULL * 1024 * 1024 * 1024, 1);     // 1GB
    testMemoryAllocation(&pool, 2ULL * 1024 * 1024 * 1024, 2);     // 2GB
    
    // Test medium allocations (should use shared memory)
    testMemoryAllocation(&pool, 4ULL * 1024 * 1024 * 1024, 3);     // 4GB
    testMemoryAllocation(&pool, 8ULL * 1024 * 1024 * 1024, 4);     // 8GB
    
    // Test large allocations (should use shared memory)
    testMemoryAllocation(&pool, 12ULL * 1024 * 1024 * 1024, 5);    // 12GB
    testMemoryAllocation(&pool, 14ULL * 1024 * 1024 * 1024, 6);    // 14GB
    
    // Test allocation that should fail
    testMemoryAllocation(&pool, 15ULL * 1024 * 1024 * 1024, 7);    // 15GB (should fail)
    
    // Test concurrent allocations
    printf("\nTesting concurrent allocations...\n");
    const int num_concurrent = 4;
    void* ptrs[num_concurrent];
    size_t sizes[num_concurrent] = {
        2ULL * 1024 * 1024 * 1024,    // 2GB (dedicated)
        4ULL * 1024 * 1024 * 1024,    // 4GB (shared)
        6ULL * 1024 * 1024 * 1024,    // 6GB (shared)
        2ULL * 1024 * 1024 * 1024     // 2GB (shared)
    };
    
    // Allocate memory blocks
    for (int i = 0; i < num_concurrent; i++) {
        err = allocateMemory(&pool, &ptrs[i], sizes[i]);
        if (err != cudaSuccess) {
            printf("Failed to allocate concurrent block %d\n", i);
            continue;
        }
        printf("Allocated block %d: %zu GB\n", i, sizes[i]/(1024*1024*1024));
        
        size_t free_size, largest_block;
        getMemoryInfo(&pool, &free_size, &largest_block);
    }
    
    // Process blocks concurrently
    for (int i = 0; i < num_concurrent; i++) {
        if (ptrs[i]) {  // Only process if allocation succeeded
            err = processData(&pool, ptrs[i], ptrs[i], sizes[i]);
            if (err != cudaSuccess) {
                printf("Failed to process block %d: %s\n", i, cudaGetErrorString(err));
            }
        }
    }
    cudaDeviceSynchronize();
    
    // Free memory blocks
    for (int i = 0; i < num_concurrent; i++) {
        if (ptrs[i]) {  // Only free if allocation succeeded
            err = freeMemory(&pool, ptrs[i]);
            if (err != cudaSuccess) {
                printf("Failed to free concurrent block %d\n", i);
                continue;
            }
            printf("Freed block %d\n", i);
        }
    }
    
    // Cleanup
    err = cleanupMemoryPool(&pool);
    if (err != cudaSuccess) {
        printf("Failed to cleanup memory pool: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\nAll tests completed!\n");
    return 0;
}
