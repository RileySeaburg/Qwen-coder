#!/bin/bash
export PATH=/usr/bin:/usr/local/cuda-12.3/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.3
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# Create symlinks for required tools
sudo ln -sf /usr/bin/gcc /usr/local/cuda-12.3/bin/gcc
sudo ln -sf /usr/bin/g++ /usr/local/cuda-12.3/bin/g++
sudo ln -sf /usr/bin/as /usr/local/cuda-12.3/bin/as

# Clean and create build directory
rm -rf build
mkdir -p build
cd build

# Run CMake
cmake .. \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/make \
  -DCUDA_HOST_COMPILER=/usr/bin/gcc \
  -DCMAKE_CUDA_ARCHITECTURES=86

# Build and install
make -j$(nproc)
sudo make install
