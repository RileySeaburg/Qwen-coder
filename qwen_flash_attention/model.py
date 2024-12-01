from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)

class QwenModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B"):
        print(f"Loading {model_name}...")
        self.model_name = model_name
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with optimizations for 32B model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",  # Let accelerate handle device mapping
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,  # Enable Flash Attention 2
            use_cache=True  # Enable KV cache
        )
        
        # Enable model parallelism if needed
        if hasattr(self.model, "enable_model_parallel"):
            self.model.enable_model_parallel()
        
        # Configure generation settings
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = True
            self.model.generation_config.top_k = 50
            self.model.generation_config.top_p = 0.95
            self.model.generation_config.repetition_penalty = 1.1
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            self.model.generation_config.max_new_tokens = 2048
        
        print("Model loaded successfully!")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate text from messages."""
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            name = msg.get("name", role)

            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "tool":
                prompt += f"Tool ({name}): {content}\n"

        # Add assistant prefix for the response
        prompt += "Assistant: "

        # Tokenize input with padding
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # Leave room for response
        )
        
        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with memory optimizations
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode output and remove prompt
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        return response

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from model's hidden states."""
        # Tokenize with padding
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate embeddings with memory optimizations
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                return_dict=True
            )

        # Get embeddings from last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = (last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        
        # Move to CPU and normalize
        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        return normalized_embeddings

    def encode(self, texts: List[str]) -> np.ndarray:
        """Alias for get_embeddings to match sentence-transformers interface."""
        return self.get_embeddings(texts)
