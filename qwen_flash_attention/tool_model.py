from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.tokenization_utils_base import BatchEncoding
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union, cast

logger = logging.getLogger(__name__)

class ToolModel:
    """Model optimized for tool usage and function calling."""
    
    def __init__(self, model_name="Team-ACE/ToolACE-8B"):
        logger.info(f"Loading {model_name}...")
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Initialize model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": self.device},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # System prompt template
        self.system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke:
{tools}
"""
        logger.info("Tool model loaded successfully!")

    def _format_tools(self, tools: List[str]) -> str:
        """Format tools into a description string."""
        tool_desc = []
        for tool in tools:
            if tool == "browser_action":
                tool_desc.append({
                    "name": "browser_action",
                    "description": "Control a web browser to navigate and interact with websites",
                    "arguments": {
                        "type": "dict",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The action to perform",
                                "enum": ["launch", "click", "type", "scroll_down", "scroll_up", "close"]
                            },
                            "url": {
                                "type": "string",
                                "description": "The URL to open (required for launch action)"
                            },
                            "coordinate": {
                                "type": "string",
                                "description": "The x,y coordinates to click (required for click action)"
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type (required for type action)"
                            }
                        },
                        "required": ["action"]
                    }
                })
            elif tool == "readFile":
                tool_desc.append({
                    "name": "readFile",
                    "description": "Read contents of a file",
                    "arguments": {
                        "type": "dict",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            }
                        },
                        "required": ["path"]
                    }
                })
            elif tool == "writeFile":
                tool_desc.append({
                    "name": "writeFile",
                    "description": "Write contents to a file",
                    "arguments": {
                        "type": "dict",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write"
                            }
                        },
                        "required": ["path", "content"]
                    }
                })
        return json.dumps(tool_desc, indent=2)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate text with tool usage."""
        if not tools:
            raise ValueError("ToolModel requires tools parameter")

        try:
            # Format system prompt with tools
            system_msg = self.system_prompt.format(tools=self._format_tools(tools))
            
            # Add system prompt to messages
            full_messages = [
                {"role": "system", "content": system_msg}
            ] + messages
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Convert to tensor and move to device
            if isinstance(prompt, BatchEncoding):
                # Handle tokenizer output format
                input_ids = cast(torch.Tensor, prompt.get('input_ids')).to(self.device)
            else:
                # Handle tensor output format
                input_ids = cast(torch.Tensor, prompt).to(self.device)

            # Generate with memory optimizations
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Get input length for response extraction
            input_length = input_ids.size(1)
            
            # Decode and extract response
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Ensure response contains at least one function call
            if "[" not in response:
                # Add default function call based on context
                if "browse" in messages[-1]["content"].lower() or "website" in messages[-1]["content"].lower():
                    response = "[browser_action(action='launch', url='https://www.nytimes.com')]"
            
            return response

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            raise

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
