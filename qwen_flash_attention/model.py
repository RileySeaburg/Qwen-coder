from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights
from transformers import BitsAndBytesConfig

class QwenModel:
    def __init__(self):
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-32B",
            trust_remote_code=True,
            cache_dir="/mnt/models/huggingface"
        )

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
            low_cpu_mem_usage=True
        )
        
        self.model.eval()

    def generate(self, prompt, max_length=2048):
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
