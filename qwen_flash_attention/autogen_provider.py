import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Union
import uvicorn
import logging
import time
import json
import torch
from autogen_schemas import ChatCompletionRequest, AgentTeamConfig
from rag_autogen import RAGAutogenTeam

# Configure logging with custom formatter
class CustomFormatter(logging.Formatter):
    green = "\033[0;32m"
    reset = "\033[0m"
    format_str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    def format(self, record):
        formatter = logging.Formatter(self.format_str)
        record.msg = f"{self.green}[AUTOGEN] {record.msg}{self.reset}"
        return formatter.format(record)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
root_logger.addHandler(console_handler)

logger = logging.getLogger("autogen_provider")

# Store active agent teams
active_teams: Dict[str, RAGAutogenTeam] = {}

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**2,  # MB
            "memory_cached": torch.cuda.memory_reserved(0) / 1024**2  # MB
        }
    return {
        "status": "healthy",
        "active_teams": len(active_teams),
        "gpu": gpu_info
    }

@app.post("/v1/groups/create")
async def create_group(config: AgentTeamConfig):
    """Create a new group chat configuration."""
    try:
        logger.debug(f"Received group creation request: {json.dumps(config.dict(), indent=2)}")
        
        group_id = f"group_{len(active_teams)}"
        logger.info(f"Creating group {group_id}")

        # Create config list for models
        config_list = []
        for agent in config.agents:
            # Map model name
            model_name = agent.model
            logger.info(f"Using model: {model_name}")

            if "claude" in model_name.lower():
                config_list.append({
                    "model": model_name,
                    "api_key": os.getenv("CLAUDE_API_KEY"),
                    "api_type": "anthropic"
                })
            else:
                config_list.append({
                    "model": model_name,
                    "api_type": "openai",
                    "base_url": "http://localhost:8000/v1"
                })

        # Create and initialize team
        team = RAGAutogenTeam(
            config_list=config_list,
            mongodb_config={
                "url": "mongodb://localhost:27017",
                "database": "autogen_vectors",
                "collection": "team_knowledge",
                "model_name": "Qwen/Qwen2.5-Coder-3B"
            }
        )
        await team.initialize()
        active_teams[group_id] = team
        
        response = {
            "group_id": group_id,
            "config": config.dict()
        }
        logger.debug(f"Group creation response: {json.dumps(response, indent=2)}")
        return response
    except Exception as e:
        logger.error(f"Error creating group: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/groups/{group_id}/chat")
async def group_chat(group_id: str, request: ChatCompletionRequest):
    """Handle group chat interactions."""
    try:
        logger.debug(f"Received chat request for group {group_id}: {json.dumps(request.dict(), indent=2)}")
        
        if group_id not in active_teams:
            logger.error(f"Group {group_id} not found")
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

        team = active_teams[group_id]

        # Get the latest user message
        user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Process task with team
        response = await team.process_task(user_message.content)

        # Format response
        return {
            "id": f"chat_{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": response["role"],
                    "content": response["content"]
                },
                "finish_reason": "stop"
            }],
            "usage": None
        }
    except Exception as e:
        logger.error(f"Error in group chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("autogen_provider:app", host="0.0.0.0", port=8001, reload=True)
