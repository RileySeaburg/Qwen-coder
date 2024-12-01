from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import signal
import asyncio
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CUDA memory management
torch.cuda.set_per_process_memory_fraction(0.8)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

app = FastAPI(title="Qwen Model Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ModelRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2048, gt=0)

class ModelResponse(BaseModel):
    text: str
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    description: str
    architecture: str
    vocab_size: int
    max_sequence_length: int
    parameters: Dict[str, Any]

class QwenModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        logger.info(f"Loaded model: {model_name}")

    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> tuple[str, Dict[str, int]]:
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = len(inputs.input_ids[0])

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        output_tokens = len(self.tokenizer.encode(response))

        # Calculate usage
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        return response, usage

    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up model: {e}")

# Initialize model
model = QwenModel()

@app.get("/info")
async def get_model_info() -> ModelInfo:
    """Get model information."""
    return ModelInfo(
        id="qwen2.5-coder-1.5b",
        name="Qwen2.5-Coder",
        version="1.5B",
        description="Qwen 2.5 coding model with Flash Attention support",
        architecture="Transformer",
        vocab_size=model.tokenizer.vocab_size,
        max_sequence_length=2048,
        parameters={
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_hidden_layers": 24
        }
    )

@app.post("/generate")
async def generate(request: ModelRequest) -> ModelResponse:
    """Generate text from prompt."""
    try:
        response, usage = await model.generate(
            prompt=request.prompt,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2048
        )
        return ModelResponse(text=response, usage=usage)
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down model server...")
    model.cleanup()
    logger.info("Model server shutdown complete")

def run_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the server with proper signal handling."""
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    # Handle signals
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: asyncio.create_task(server.shutdown()))

    # Run server
    server.run()
