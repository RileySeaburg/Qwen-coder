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

:: Create build directory
if not exist build mkdir build
cd build

:: Configure with CMake
echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 -T cuda=12.2 ^
    -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_PATH%" ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
    ..

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

:: Build
echo Building project...
cmake --build . --config Release --verbose
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

:: Run tests
echo.
echo Running memory manager tests...
Release\test_memory_manager.exe

cd ..
echo.
echo Build completed successfully!
