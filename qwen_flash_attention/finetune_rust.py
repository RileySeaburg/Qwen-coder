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
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    cache_dir: str = "/mnt/models/huggingface"
) -> Tuple[AutoModelForCausalLM, TokenizerType]:
    """Setup the base model and tokenizer with optimized memory settings"""
    
    # Configure 4-bit quantization with double quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8
    )

    # Load tokenizer with optimized settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        trust_remote_code=True,
        model_max_length=1024  # Reduced for memory efficiency
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={0: "45GB", "cpu": "48GB"},  # Reserve some GPU memory
        offload_folder="/mnt/models/offload"
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Configure LoRA with memory-efficient settings"""
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Reduced rank for memory efficiency
        lora_alpha=16,
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
        ],
        modules_to_save=["embed_tokens", "lm_head"]  # Save important layers
    )
    
    model = get_peft_model(model, lora_config)
    return model

def prepare_rust_dataset(
    tokenizer: TokenizerType,
    max_length: int = 1024,  # Reduced for memory efficiency
    batch_size: int = 1  # Small batch size
):
    """Load and prepare the Rust dataset with memory optimizations"""
    
    # Load dataset with streaming
    dataset = load_dataset(
        "ammarnasr/the-stack-rust-clean",
        split="train",
        streaming=True
    )
    
    def format_rust_code(example):
        """Format Rust code with instruction prompt"""
        instruction = (
            "You are an expert Rust programmer. Below is a Rust code snippet. "
            "Study it carefully to learn Rust programming patterns and best practices.\n\n"
            "Code:\n```rust\n{}\n```"
        ).format(example["content"][:max_length])  # Truncate long examples
        
        return {"text": instruction}
    
    # Format dataset with efficient mapping
    dataset = dataset.map(
        format_rust_code,
        remove_columns=dataset.column_names,
        num_proc=4  # Parallel processing
    )
    
    def tokenize_function(examples):
        """Memory-efficient tokenization"""
        outputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return outputs
    
    # Tokenize dataset with efficient batching
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=4
    )
    
    return tokenized_dataset

def train(
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    cache_dir: str = "/mnt/models/huggingface",
    output_dir: str = "/mnt/models/rust-llama",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,  # Small batch size
    gradient_accumulation_steps: int = 32,  # Increased for stability
    learning_rate: float = 1e-4,  # Reduced for stability
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    save_steps: int = 100,
    logging_steps: int = 10,
    max_steps: int = 10000  # Limit training time
):
    """Main training function with memory optimizations"""
    
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, cache_dir)
    
    logger.info("Setting up LoRA...")
    model = setup_lora(model)
    
    logger.info("Preparing dataset...")
    dataset = prepare_rust_dataset(tokenizer=tokenizer)
    
    # Setup training arguments with memory optimizations
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
        save_total_limit=2,  # Keep fewer checkpoints
        push_to_hub=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,
        max_steps=max_steps,
        optim="adamw_torch_fused",  # Use fused optimizer
        dataloader_num_workers=4,
        group_by_length=True,  # Group similar lengths for efficiency
        ignore_data_skip=True,
        ddp_find_unused_parameters=False,
        torch_compile=True,  # Enable torch.compile
        use_cpu=False
    )
    
    # Setup data collator with memory efficiency
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for hardware
    )
    
    # Initialize trainer with memory optimizations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Enable mixed precision training
    trainer.accelerator.state.use_fp16 = True
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving final model...")
    trainer.save_model()
    
    return model, tokenizer

if __name__ == "__main__":
    # Set environment variables for optimized training
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        model, tokenizer = train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
