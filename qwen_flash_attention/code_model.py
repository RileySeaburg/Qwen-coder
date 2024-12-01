import logging
import torch
from torch import Tensor
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)
from transformers.generation.utils import GenerateOutput
from typing import List, Optional, Dict, Any, Union, TypedDict, cast
import numpy as np
from .cuda_utils import CUDAManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("code_model")

class ModelKwargs(TypedDict):
    device_map: str
    torch_dtype: torch.dtype
    trust_remote_code: bool
    max_memory: Dict[int, str]
    low_cpu_mem_usage: bool
    output_hidden_states: bool
    return_dict_in_generate: bool

class TokenizerOutput(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor

class QwenForCausalLM:
    _instance: Optional['QwenForCausalLM'] = None
    _model: Optional[PreTrainedModel] = None
    _tokenizer: Optional[PreTrainedTokenizer] = None
    _initialized: bool = False
    _cuda_manager: Optional[CUDAManager] = None

    def __new__(cls) -> 'QwenForCausalLM':
        if cls._instance is None:
            cls._instance = super(QwenForCausalLM, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not QwenForCausalLM._initialized:
            self.model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
            if torch.cuda.is_available():
                QwenForCausalLM._cuda_manager = CUDAManager(0)  # Use first GPU
            self._initialize()
            QwenForCausalLM._initialized = True

    def _initialize(self) -> None:
        """Initialize model and tokenizer if not already initialized."""
        try:
            if QwenForCausalLM._model is None:
                logger.info(f"Loading model: {self.model_name}")
                # Use CUDA manager to optimize memory
                if self._cuda_manager:
                    self._cuda_manager.optimize_memory()

                model_kwargs: ModelKwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                    "max_memory": {0: "3GiB"},  # Limit GPU memory usage
                    "low_cpu_mem_usage": True,
                    "output_hidden_states": True,  # Enable hidden states output
                    "return_dict_in_generate": True  # Enable dict return in generate
                }

                QwenForCausalLM._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                logger.info("Model loaded successfully")
            else:
                logger.info("Using existing model instance")

            if QwenForCausalLM._tokenizer is None:
                logger.info(f"Initializing tokenizer for model: {self.model_name}")
                QwenForCausalLM._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    pad_token='<|extra_0|>',
                    eos_token='<|endoftext|>'
                )
                logger.info("Tokenizer initialized successfully")
            else:
                logger.info("Using existing tokenizer instance")

        except Exception as e:
            logger.error(f"Error in initialization: {e}")
            raise

    @property
    def model(self) -> PreTrainedModel:
        """Get the model instance."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer instance."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self._tokenizer

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        prompt = "System: You are a professional programmer. Respond in plain text only. No emojis, hashtags, or social media style content. Keep responses minimal and focused on the technical task.\n\n"
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"Instructions: {msg['content']}\n\n"
            else:
                name = msg.get("name", msg["role"])
                prompt += f"{name}: {msg['content']}\n"

        # Add assistant prefix for the response
        prompt += "assistant: "
        
        return prompt

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from model's hidden states."""
        try:
            # Ensure model and tokenizer are initialized
            if not self._initialized:
                raise RuntimeError("Model not initialized")
            
            # Get model and tokenizer
            model = self.model
            tokenizer = self.tokenizer

            # Tokenize all texts
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            # Forward pass with output_hidden_states=True
            with torch.inference_mode():
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Mean pool over sequence length to get embeddings
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = (last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

            # Convert to numpy and normalize
            embeddings = embeddings.cpu().numpy()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms

            return normalized_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """Alias for get_embeddings to match sentence-transformers interface."""
        return self.get_embeddings(texts)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> str:
        """Generate text using the model."""
        try:
            logger.debug(f"Generating response for {len(messages)} messages")
            
            # Ensure model and tokenizer are initialized
            if not self._initialized:
                raise RuntimeError("Model not initialized")
            
            # Get model and tokenizer (will raise if not initialized)
            model = self.model
            tokenizer = self.tokenizer
            
            # Optimize memory before generation
            if self._cuda_manager:
                self._cuda_manager.optimize_memory()

            # Format messages into prompt
            prompt = self._format_prompt(messages)
            logger.debug(f"Generated prompt: {prompt}")

            try:
                # Tokenize input with conservative limits
                inputs: TokenizerOutput = cast(TokenizerOutput, tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ))
                
                # Use CUDA manager to prepare input
                if self._cuda_manager:
                    input_ids = self._cuda_manager.prepare_input(inputs["input_ids"])
                    attention_mask = self._cuda_manager.prepare_input(inputs["attention_mask"])
                else:
                    input_ids = inputs["input_ids"].to(model.device)
                    attention_mask = inputs["attention_mask"].to(model.device)
                
                logger.debug("Input tokenized successfully")

                # Create generation config
                generation_config = GenerationConfig(
                    max_new_tokens=256,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )

                # Generate text with conservative settings
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config
                    )
                logger.debug("Text generated successfully")

                # Decode output
                generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                logger.debug(f"Full generated text: {generated_text}")
                
                # Remove the prompt from the generated text
                response = generated_text[len(prompt):].strip()
                logger.debug(f"Final response: {response}")

                # Optimize memory after generation
                if self._cuda_manager:
                    self._cuda_manager.optimize_memory()
                
                return response

            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error during generation: {e}")
                    if self._cuda_manager:
                        self._cuda_manager.optimize_memory()
                    return "Error: GPU memory issue occurred. Please try again with a shorter input."
                raise

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            raise
