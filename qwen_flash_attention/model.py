from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights
from transformers import BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

class QwenModel:
    def __init__(self):
        try:
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Configure device map and memory
            max_memory = {
                0: "12GB",  # GPU - reduced from previous value
                "cpu": "24GB"  # CPU RAM - reduced from previous value
            }

            logger.info("Loading Qwen tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B",  # Using smaller model
                trust_remote_code=True
            )

            logger.info("Loading Qwen model...")
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B",  # Using smaller model
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )
            
            self.model.eval()
            logger.info("Qwen model loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing Qwen model: {str(e)}")
            raise

    def generate(self, prompt, max_length=2048):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
