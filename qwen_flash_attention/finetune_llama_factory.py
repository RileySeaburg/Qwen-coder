import os
import json
from llmtuner import BaseTrainer, DataArguments, FinetuningArguments, ModelArguments

def prepare_dataset():
    """Prepare the dataset in LLaMA Factory format"""
    dataset_path = "rust_dataset.json"
    
    # Create dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        from datasets import load_dataset
        
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
    
    # Configure arguments
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen-7B",
        use_flash_attn=True
    )
    
    data_args = DataArguments(
        dataset=dataset_path,
        dataset_format="sharegpt"
    )
    
    training_args = FinetuningArguments(
        output_dir="output",
        overwrite_cache=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-4,
        num_train_epochs=3,
        bf16=True,
        quantization_bit=4,
        lora_target="c_attn,c_proj,w1,w2",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    
    # Initialize trainer
    trainer = BaseTrainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    
    # Run training
    trainer.train()
    
    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()
