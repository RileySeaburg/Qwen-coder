import os
import json
import subprocess
from datasets import load_dataset

def prepare_dataset():
    """Prepare the dataset in LLaMA Factory format"""
    dataset_path = "data/rust_dataset.json"
    os.makedirs("data", exist_ok=True)
    
    # Create dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        # Load Rust dataset
        dataset = load_dataset("ammarnasr/the-stack-rust-clean", split="train[:10000]")
        
        # Convert to LLaMA Factory format
        formatted_data = []
        for idx, item in enumerate(dataset):
            formatted_data.append({
                "conversation": [
                    {
                        "system": "You are an expert Rust programmer. Study the following code carefully to learn Rust programming patterns and best practices.",
                        "input": f"Here is a Rust code snippet:\n```rust\n{item['content']}\n```",
                        "output": "I will analyze this code and learn from its patterns and practices."
                    }
                ]
            })
        
        # Save dataset
        with open(dataset_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
    
    return dataset_path

def main():
    # Prepare dataset
    dataset_path = prepare_dataset()
    
    # Run LLaMA Factory CLI
    subprocess.run([
        "python3",
        "LLaMA-Factory/src/entry_point.py",
        "train",
        "--model_name_or_path", "Qwen/Qwen-7B",
        "--dataset", dataset_path,
        "--dataset_format", "sharegpt",
        "--output_dir", "output",
        "--overwrite_cache",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "32",
        "--learning_rate", "1e-4",
        "--num_train_epochs", "3",
        "--bf16",
        "--quantization_bit", "4",
        "--lora_target", "c_attn,c_proj,w1,w2",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--flash_attn",
        "--shift_attn",
        "--rope_scaling", "dynamic",
        "--rope_factor", "2.0"
    ], check=True)

if __name__ == "__main__":
    main()
