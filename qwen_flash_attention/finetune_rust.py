import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import logging
from typing import Dict, Sequence, Union, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

def setup_model_and_tokenizer(
    model_name: str = "Qwen/Qwen-7B",
    cache_dir: str = "/mnt/models/huggingface"
) -> Tuple[AutoModelForCausalLM, TokenizerType]:
    """Setup the base model and tokenizer with optimized memory settings"""
    
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
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model - Qwen handles flash attention internally
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Configure LoRA with memory-efficient settings"""
    
    # Updated target modules for Qwen architecture
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Reduced rank for memory efficiency
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "c_attn",
            "c_proj",
            "w1",
            "w2"
        ],
        modules_to_save=["wte", "ln_f"]  # Save important layers
    )
    
    model = get_peft_model(model, lora_config)
    return model

def prepare_rust_dataset(
    tokenizer: TokenizerType,
    max_length: int = 2048,
    max_samples: int = 10000  # Limit dataset size for memory efficiency
):
    """Load and prepare the Rust dataset"""
    
    # Load dataset without streaming for better memory efficiency
    dataset = load_dataset(
        "ammarnasr/the-stack-rust-clean",
        split=f"train[:{max_samples}]"  # Take first max_samples examples
    )
    
    def format_rust_code(examples):
        """Format Rust code with Qwen chat template"""
        instructions = [
            "<|im_start|>system\nYou are an expert Rust programmer. Study the following code carefully to learn Rust programming patterns and best practices.<|im_end|>\n"
            "<|im_start|>user\nHere is a Rust code snippet:\n```rust\n{}\n```<|im_end|>\n"
            "<|im_start|>assistant\nI will analyze this code and learn from its patterns and practices.<|im_end|>\n"
            .format(content[:max_length])  # Truncate long examples
            for content in examples["content"]
        ]
        
        return {"text": instructions}
    
    # Format and tokenize dataset
    dataset = dataset.map(
        format_rust_code,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting code"
    )
    
    def tokenize_function(examples):
        """Tokenize the text"""
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def train(
    model_name: str = "Qwen/Qwen-7B",
    cache_dir: str = "/mnt/models/huggingface",
    output_dir: str = "/mnt/models/rust-qwen",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    save_steps: int = 100,
    logging_steps: int = 10,
    max_steps: int = 10000
):
    """Main training function"""
    
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, cache_dir)
    
    logger.info("Setting up LoRA...")
    model = setup_lora(model)
    
    logger.info("Preparing dataset...")
    dataset = prepare_rust_dataset(tokenizer=tokenizer)
    
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
        save_total_limit=2,
        push_to_hub=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,
        max_steps=max_steps,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        group_by_length=True,  # Now works since we're not using streaming
        ignore_data_skip=True,
        ddp_find_unused_parameters=False
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
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
    try:
        model, tokenizer = train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
