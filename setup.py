from setuptools import setup, find_packages

setup(
    name="qwen_flash_attention",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "httpx>=0.24.1",
        "pydantic>=2.0.0",
        "einops>=0.6.1",
        "pymongo>=4.3.3",
        "numpy>=1.24.3",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.1",
        "scipy>=1.10.1",
        "sentencepiece>=0.1.99",
        "protobuf>=4.24.4",
        "tiktoken>=0.5.1",
        "anthropic>=0.8.1",
        "ninja",  # Required for flash-attention compilation
        "packaging"  # Required for flash-attention
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy"
        ]
    },
    python_requires=">=3.10",
)
