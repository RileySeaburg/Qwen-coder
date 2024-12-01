@echo off
setlocal

:: Check for CUDA installation
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo CUDA toolkit not found! Please install CUDA toolkit.
    exit /b 1
)

:: Print CUDA version
nvcc --version

:: Set paths
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64

:: Compile test program with memory manager
echo.
echo Compiling memory manager test program...
nvcc -O3 -arch=sm_75 -Xcompiler "/MD" ^
    -o test_memory_manager.exe ^
    memory_manager.cu test_memory_manager.cu ^
    -L"%CUDA_LIB_PATH%" -lcudart ^
    -lineinfo -Xptxas=-v

if %ERRORLEVEL% neq 0 (
    echo Failed to compile!
    exit /b 1
)

:: Run test
echo.
echo Running memory manager tests...
test_memory_manager.exe

echo.
echo Build completed successfully!
