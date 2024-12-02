import logging
import json
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from .browser_session import BrowserSession

logger = logging.getLogger(__name__)

class ToolModel:
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
                0: "20GB",  # GPU
                "cpu": "48GB"  # CPU RAM
            }

            logger.info("Loading ToolACE tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Team-ACE/ToolACE-8B",
                trust_remote_code=True,
                cache_dir="/mnt/models/huggingface"
            )

            logger.info("Loading ToolACE model...")
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                "Team-ACE/ToolACE-8B",
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
            logger.info("ToolACE model loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing ToolACE model: {str(e)}")
            raise

    async def process_browser_action(self, action_json):
        try:
            action_data = json.loads(action_json)
            if action_data.get("action") == "browse" and action_data.get("url"):
                async with BrowserSession("tool_agent") as browser:
                    await browser.launch(action_data["url"])
                    # Wait a moment for page to load
                    await asyncio.sleep(2)
                    # Get the current state including screenshot
                    state = browser.get_state()
                    await browser.close()
                    return f"Browsed to {action_data['url']}. Current state: {json.dumps(state)}"
            return "Invalid browser action format"
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in browser action")
            return "Invalid JSON format in browser action"
        except Exception as e:
            logger.error(f"Error processing browser action: {str(e)}")
            return f"Error processing browser action: {str(e)}"

    def generate(self, prompt, max_length=2048):
        try:
            full_prompt = (
                "You are a helpful AI assistant that can use tools to accomplish tasks. "
                "You have access to a web browser tool.\n\n"
                "When you need to browse the web, respond with a JSON object containing:\n"
                '{"action": "browse", "url": "the_url_to_visit"}\n\n'
                "Task: " + prompt
            )

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if response contains a browser action
            try:
                if '{"action": "browse"' in response:
                    # Extract the JSON part
                    start = response.find('{')
                    end = response.find('}', start) + 1
                    action_json = response[start:end]
                    # Process the browser action
                    return asyncio.run(self.process_browser_action(action_json))
                return response
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
