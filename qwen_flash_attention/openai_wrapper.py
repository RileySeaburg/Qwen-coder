from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from .code_model import QwenForCausalLM
import logging
import json
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize model (singleton)
model = QwenForCausalLM()

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    agent_config: Optional[Dict[str, Any]] = None

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Convert messages to list of dicts for the model
        messages = [msg.dict() for msg in request.messages]
        
        response = await model.generate(
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2048
        )

        return {
            "id": "chat-completion",
            "object": "chat.completion",
            "created": None,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "data": [
            {
                "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "object": "model",
                "created": None,
                "owned_by": "Qwen",
                "permission": [],
                "root": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "parent": None,
                "capabilities": {
                    "chat_completion": True
                }
            }
        ],
        "object": "list"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
