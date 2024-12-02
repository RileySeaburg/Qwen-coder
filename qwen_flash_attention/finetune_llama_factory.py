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
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are an expert Rust programmer. Study the following code carefully to learn Rust programming patterns and best practices."
                    },
                    {
                        "from": "human",
                        "value": f"Here is a Rust code snippet:\n```rust\n{item['content']}\n```"
                    },
                    {
                        "from": "assistant",
                        "value": "I will analyze this code and learn from its patterns and practices."
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
    
    # Launch LLaMA Factory WebUI
    env = os.environ.copy()
    env["GRADIO_SHARE"] = "1"  # Create a public URL
    
    subprocess.run([
        "python3",
        "-m", "llmtuner.webui",
        "--share"
    ], env=env, check=True)

if __name__ == "__main__":
    main()
