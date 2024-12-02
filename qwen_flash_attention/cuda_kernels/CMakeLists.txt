cmake_minimum_required(VERSION 3.18)

# Set CUDA host compiler before project declaration
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc")
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set CUDA paths explicitly
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.3")
set(CUDA_BIN_PATH "/usr/local/cuda-12.3/bin")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-12.3/include")
set(CUDA_TOOLKIT_INCLUDE "/usr/local/cuda-12.3/include")
set(CMAKE_PREFIX_PATH "/usr/local/cuda-12.3")

project(memory_manager CUDA CXX)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
include(FindCUDA)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please specify CUDA_TOOLKIT_ROOT_DIR.")
endif()

include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA flags for better memory management
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_86,code=sm_86
    --use_fast_math
    --ptxas-options=-v
    -lineinfo
    -Xcompiler -fPIC
    --default-stream per-thread
    --maxrregcount=128
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
