#!/bin/bash

# Exit on error
set -e

echo "Setting up fine-tuning environment..."

# Create directories if they don't exist
sudo mkdir -p /mnt/models/rust-llama
sudo chown -R ubuntu:ubuntu /mnt/models/rust-llama

# Install dependencies
echo "Installing dependencies..."
pip install -r finetune_requirements.txt

# Set up environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/mnt/models/huggingface
export HF_HOME=/mnt/models/huggingface
export WANDB_DISABLED=true

# Clean up any existing cached models (optional)
echo "Cleaning up old model files..."
rm -rf /mnt/models/huggingface/models--meta-llama--Llama-3.2-11B-Vision-Instruct 2>/dev/null || true

# Start training
echo "Starting fine-tuning process..."
python finetune_rust.py 2>&1 | tee finetune.log

echo "Fine-tuning complete! Check finetune.log for details."
