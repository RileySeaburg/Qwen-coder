cmake_minimum_required(VERSION 3.18)
project(memory_manager CUDA CXX)

# Set CUDA version
set(CMAKE_CUDA_COMPILER_VERSION "12.2")

# Set CUDA architecture based on RTX 2080 SUPER (Turing)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA 12.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_75,code=sm_75
    --use_fast_math
    --ptxas-options=-v
    -lineinfo
)

# Add library
cuda_add_library(memory_manager SHARED
    memory_manager.cu
)

# Add test executable
cuda_add_executable(test_memory_manager
    test_memory_manager.cu
)

# Set properties
set_target_properties(memory_manager PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(test_memory_manager PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
target_link_libraries(memory_manager
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_cudart_static_LIBRARY}
)

target_link_libraries(test_memory_manager
    memory_manager
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_cudart_static_LIBRARY}
)

# Install rules
install(TARGETS memory_manager test_memory_manager
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES memory_manager.cuh
    DESTINATION include)
