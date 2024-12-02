import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import logging
from typing import Dict, Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    cache_dir: str = "/mnt/models/huggingface"
):
    """Setup the base model and tokenizer with quantization"""
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA adapters for efficient fine-tuning"""
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )
    
    model = get_peft_model(model, lora_config)
    return model

def prepare_rust_dataset(
    tokenizer,
    max_length: int = 2048,
    batch_size: int = 4
):
    """Load and prepare the Rust dataset"""
    
    dataset = load_dataset("ammarnasr/the-stack-rust-clean", split="train")
    
    def format_rust_code(example):
        """Format Rust code with instruction prompt"""
        instruction = (
            "You are an expert Rust programmer. Below is a Rust code snippet. "
            "Study it carefully to learn Rust programming patterns and best practices.\n\n"
            "Code:\n```rust\n{}\n```"
        ).format(example["content"])
        
        return {"text": instruction}
    
    # Format dataset
    dataset = dataset.map(format_rust_code, remove_columns=dataset.column_names)
    
    def tokenize_function(examples):
        """Tokenize the text with proper padding and truncation"""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def train(
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    cache_dir: str = "/mnt/models/huggingface",
    output_dir: str = "/mnt/models/rust-llama",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    save_steps: int = 100,
    logging_steps: int = 10
):
    """Main training function"""
    
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, cache_dir)
    
    logger.info("Setting up LoRA...")
    model = setup_lora(model)
    
    logger.info("Preparing dataset...")
    dataset = prepare_rust_dataset(tokenizer)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=3,
        push_to_hub=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving final model...")
    trainer.save_model()
    
    return model, tokenizer

if __name__ == "__main__":
    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    try:
        model, tokenizer = train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
