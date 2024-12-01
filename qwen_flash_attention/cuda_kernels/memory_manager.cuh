#ifndef MEMORY_MANAGER_CUH
#define MEMORY_MANAGER_CUH

#include <cstddef>

// Initialize the memory manager with a specified pool size
bool initializeMemoryManager(size_t size);

// Allocate memory from the pool
void* allocateMemory(size_t size);

// Free previously allocated memory
bool freeMemory(void* ptr);

// Shutdown the memory manager and free all resources
bool shutdownMemoryManager();

// Get the amount of available memory in the pool
size_t getAvailableMemory();

// Defragment the memory pool to reduce fragmentation
bool defragmentMemory();

#endif // MEMORY_MANAGER_CUH
