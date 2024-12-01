import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

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
                0: "25GB",  # GPU
                "cpu": "48GB"  # CPU RAM
            }

            logger.info("Loading Qwen tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-32B",
                trust_remote_code=True,
                cache_dir="/mnt/models/huggingface"
            )

            logger.info("Loading Qwen model...")
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-32B",
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                max_memory=max_memory,
                offload_folder="/mnt/models/offload",
                trust_remote_code=True,
                cache_dir="/mnt/models/huggingface",
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
