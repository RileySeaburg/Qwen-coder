import logging
import json
import asyncio
import torch
from typing import Optional, Dict, List, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from .browser_session import BrowserSession
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class ToolModel:
    def __init__(self, model_path: str = "meta-llama/Meta-Llama-3-8B", 
                 cache_dir: str = "/mnt/models/huggingface",
                 max_gpu_memory: str = "20GB",
                 max_cpu_memory: str = "48GB",
                 timeout: int = 30):
        """
        Initialize the ToolModel with configurable parameters
        
        Args:
            model_path: Path to the model
            cache_dir: Directory for model cache
            max_gpu_memory: Maximum GPU memory to use
            max_cpu_memory: Maximum CPU memory to use
            timeout: Default timeout for operations in seconds
        """
        try:
            self.timeout = timeout
            self._validate_init_params(model_path, cache_dir, max_gpu_memory, max_cpu_memory)
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Configure device map and memory
            max_memory = {
                0: max_gpu_memory,
                "cpu": max_cpu_memory
            }

            logger.info("Loading LLaMA tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=cache_dir,
                padding_side="left",
                bos_token="<|begin_of_text|>",
                eos_token="<|end_of_text|>",
                model_max_length=2048
            )
            # Set pad token to eos token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            logger.info("Loading LLaMA model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                max_memory=max_memory,
                offload_folder="/mnt/models/offload",
                trust_remote_code=True,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            logger.info("LLaMA model loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing LLaMA model: {str(e)}")
            raise

    def _validate_init_params(self, model_path: str, cache_dir: str, 
                            max_gpu_memory: str, max_cpu_memory: str) -> None:
        """Validate initialization parameters"""
        if not isinstance(model_path, str) or not model_path:
            raise ValidationError("Invalid model_path")
        if not isinstance(cache_dir, str) or not cache_dir:
            raise ValidationError("Invalid cache_dir")
        if not isinstance(max_gpu_memory, str) or not max_gpu_memory.endswith(("GB", "MB")):
            raise ValidationError("Invalid max_gpu_memory format")
        if not isinstance(max_cpu_memory, str) or not max_cpu_memory.endswith(("GB", "MB")):
            raise ValidationError("Invalid max_cpu_memory format")

    async def process_browser_action(self, action_json: str, timeout: Optional[int] = None) -> str:
        """
        Process browser-specific actions with timeout and validation
        
        Args:
            action_json: JSON string containing browser action
            timeout: Optional timeout override
        
        Returns:
            String response from browser action
        """
        timeout = timeout or self.timeout
        try:
            # Validate action JSON
            if not isinstance(action_json, str):
                raise ValidationError("action_json must be a string")
            
            action_data = json.loads(action_json)
            
            # Validate required fields
            if not isinstance(action_data, dict):
                raise ValidationError("Invalid action format")
            if "action" not in action_data or "url" not in action_data:
                raise ValidationError("Missing required fields")
            if not isinstance(action_data["url"], str):
                raise ValidationError("URL must be a string")
                
            if action_data.get("action") == "browse":
                try:
                    async with asyncio.timeout(timeout):
                        async with BrowserSession("tool_agent") as browser:
                            await browser.launch(action_data["url"])
                            # Wait for page load with timeout
                            await asyncio.sleep(min(2, timeout/2))
                            state = browser.get_state()
                            await browser.close()
                            return f"Browsed to {action_data['url']}. Current state: {json.dumps(state)}"
                except asyncio.TimeoutError:
                    logger.error("Browser action timed out")
                    return "Browser action timed out"
                
            return "Invalid browser action format"
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in browser action: {str(e)}")
            return "Invalid JSON format in browser action"
        except ValidationError as e:
            logger.error(f"Validation error in browser action: {str(e)}")
            return f"Validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Error processing browser action: {str(e)}")
            return f"Error processing browser action: {str(e)}"

    def _validate_function_call(self, func_call: Dict[str, Any]) -> None:
        """Validate a single function call"""
        if not isinstance(func_call, dict):
            raise ValidationError("Function call must be a dictionary")
        if "name" not in func_call:
            raise ValidationError("Function call missing name")
        if "parameters" not in func_call:
            raise ValidationError("Function call missing parameters")
        if not isinstance(func_call["parameters"], dict):
            raise ValidationError("Parameters must be a dictionary")

    async def process_function_calls(self, response: str, timeout: Optional[int] = None) -> str:
        """
        Process single, parallel, or dependent function calls with validation
        
        Args:
            response: Model response containing function calls
            timeout: Optional timeout override
        
        Returns:
            JSON string of processed function calls
        """
        timeout = timeout or self.timeout
        try:
            if not isinstance(response, str):
                raise ValidationError("Response must be a string")

            # Extract function calls
            function_calls: List[Dict[str, Any]] = []
            lines = response.split('\n')
            
            try:
                async with asyncio.timeout(timeout):
                    for line in lines:
                        if line.startswith('[') and line.endswith(']'):
                            # Parse function calls
                            calls = line.strip('[]').split(',')
                            for call in calls:
                                call = call.strip()
                                if '(' in call and ')' in call:
                                    func_name = call[:call.find('(')]
                                    params_str = call[call.find('(')+1:call.find(')')]
                                    params = {}
                                    
                                    # Parse parameters with validation
                                    for param in params_str.split(','):
                                        if '=' in param:
                                            key, value = param.split('=')
                                            key = key.strip()
                                            value = value.strip().strip('"\'')
                                            
                                            # Basic parameter validation
                                            if not key:
                                                raise ValidationError("Empty parameter name")
                                            if not value and value != "0":
                                                raise ValidationError(f"Empty value for parameter {key}")
                                                
                                            params[key] = value
                                    
                                    func_call = {
                                        'name': func_name.strip(),
                                        'parameters': params
                                    }
                                    
                                    # Validate function call
                                    self._validate_function_call(func_call)
                                    function_calls.append(func_call)

            except asyncio.TimeoutError:
                logger.error("Function call processing timed out")
                return "Function call processing timed out"

            # Handle browser actions specially
            for call in function_calls:
                if call['name'] == 'browse':
                    action_json = json.dumps({
                        'action': 'browse',
                        'url': call['parameters'].get('url', '')
                    })
                    return await self.process_browser_action(action_json, timeout)

            return json.dumps(function_calls)

        except ValidationError as e:
            logger.error(f"Validation error in function calls: {str(e)}")
            return f"Validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Error processing function calls: {str(e)}")
            return f"Error processing function calls: {str(e)}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, max_length: int = 2048, 
                temperature: float = 0.7, timeout: Optional[int] = None) -> str:
        """
        Generate response with enhanced function calling capabilities and retries
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            timeout: Optional timeout override
        
        Returns:
            Generated response
        """
        timeout = timeout or self.timeout
        try:
            # Validate input parameters
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValidationError("Invalid prompt")
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValidationError("Invalid max_length")
            if not isinstance(temperature, (int, float)) or temperature <= 0:
                raise ValidationError("Invalid temperature")

            # Enhanced prompt template with special tokens
            full_prompt = (
                self.tokenizer.bos_token +
                "You are an expert in composing functions. You are given a question and a set of possible functions. "
                "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n\n"
                "If none of the functions can be used, point it out. "
                "If the given question lacks the parameters required by the function, also point it out.\n\n"
                "You should only return the function call in tools call sections.\n\n"
                "If you decide to invoke any of the function(s), you MUST put it in the format of "
                "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
                "You SHOULD NOT include any other text in the response.\n\n"
                "Task: " + prompt + self.tokenizer.eos_token
            )

            # Tokenize with validation
            if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
                raise RuntimeError("Model or tokenizer not initialized")
                
            inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
            if not isinstance(inputs, dict) or "input_ids" not in inputs:
                raise ValidationError("Tokenization failed")
                
            inputs = inputs.to(self.model.device)

            # Generate with optimized parameters and timeout
            start_time = asyncio.get_event_loop().time()
            with torch.inference_mode():
                while asyncio.get_event_loop().time() - start_time < timeout:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    if not isinstance(outputs, torch.Tensor):
                        raise ValidationError("Invalid model output")
                        
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if not isinstance(response, str):
                        raise ValidationError("Invalid decoded response")
                    
                    # Process function calls
                    return asyncio.run(self.process_function_calls(response, timeout))
                
            logger.error("Generation timed out")
            return "Generation timed out"

        except ValidationError as e:
            logger.error(f"Validation error in generate: {str(e)}")
            return f"Validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
