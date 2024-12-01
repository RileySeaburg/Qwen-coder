#include <stdio.h>
#include <cuda_runtime.h>
#include "memory_manager.cuh"

// Test kernels
__global__ void writeTest(int* data, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void verifyTest(int* data, int size, int value, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && data[idx] != value) {
        atomicAdd(result, 1);
    }
}

// Main test function
int main() {
    printf("Starting memory manager tests...\n");

    // Initialize memory manager with 1GB
    if (!initializeMemoryManager(1ULL * 1024 * 1024 * 1024)) {
        printf("Failed to initialize memory manager\n");
        return 1;
    }

    // Test basic allocation
    size_t alloc_size = 256 * 1024 * 1024; // 256MB
    void* ptr = allocateMemory(alloc_size);
    if (!ptr) {
        printf("Failed to allocate memory\n");
        return 1;
    }

    // Test kernel execution with allocated memory
    int num_elements = alloc_size / sizeof(int);
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    int test_value = 42;

    // Write test
    writeTest<<<blocks, threads>>>((int*)ptr, num_elements, test_value);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Write kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // Verify test
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    verifyTest<<<blocks, threads>>>((int*)ptr, num_elements, test_value, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Verify kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return 1;
    }
    cudaDeviceSynchronize();

    // Check result
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    if (h_result != 0) {
        printf("Memory verification failed: %d errors found\n", h_result);
        return 1;
    }

    // Test memory info
    size_t available = getAvailableMemory();
    printf("Available memory: %zu bytes\n", available);

    // Test memory deallocation
    if (!freeMemory(ptr)) {
        printf("Failed to free memory\n");
        return 1;
    }

    // Test memory defragmentation
    if (!defragmentMemory()) {
        printf("Failed to defragment memory\n");
        return 1;
    }

    // Cleanup
    if (!shutdownMemoryManager()) {
        printf("Failed to shutdown memory manager\n");
        return 1;
    }

    printf("All tests passed successfully!\n");
    return 0;
}
