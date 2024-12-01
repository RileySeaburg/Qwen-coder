from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ToolModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Team-ACE/ToolACE-8B", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "Team-ACE/ToolACE-8B",
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "20GiB"},
            trust_remote_code=True,
            offload_folder="/mnt/models/offload"
        )
        self.model.eval()

    def generate(self, prompt, max_length=2048):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
