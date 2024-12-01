import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .model import QwenModel
from .schemas import (
    ChatCompletionRequest,    
    ChatCompletionResponse,
    Agent,
    AgentTeamConfig,
    Message,
    TaskType,
    TeamSelection
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instance
_qwen_model: Optional[QwenModel] = None

def init_model(model: QwenModel):
    """Initialize global model instance."""
    global _qwen_model
    _qwen_model = model

def get_model() -> QwenModel:
    """Get global model instance."""
    if _qwen_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    return _qwen_model

class AgentResponse(BaseModel):
    """Response from an agent."""
    role: str
    content: str
    name: str
    metadata: Optional[Dict[str, Any]] = None

def message_to_dict(msg: Message) -> Dict[str, str]:
    """Convert Message object to dictionary."""
    return {
        "role": msg.role,
        "content": msg.content,
        "name": msg.name if msg.name else ""
    }

async def route_to_agent(
    request: ChatCompletionRequest,
    agent: Agent,
    task_type: TaskType = "chat",
    team_selection: TeamSelection = "default"
) -> ChatCompletionResponse:
    """Route request to appropriate agent."""
    try:
        model = get_model()
        
        # Add agent system prompt to messages
        messages = [
            message_to_dict(Message(role="system", content=agent.systemPrompt))
        ] + [message_to_dict(msg) for msg in request.messages]
        
        # Generate response
        response = await model.generate(
            messages=messages,
            temperature=request.temperature or 0.7,  # Default temperature
            max_tokens=request.max_tokens or 1024,  # Default max tokens
            tools=request.tools
        )
        
        return ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677858242,
            model=request.model or agent.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                    "name": agent.name
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
        
    except Exception as e:
        logger.error(f"Error routing to agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/agent/chat/completions", response_model=ChatCompletionResponse)
async def agent_chat_completion(request: ChatCompletionRequest, agent: Agent):
    """Handle agent chat completion requests."""
    return await route_to_agent(request, agent)

@router.post("/v1/agent/team/chat", response_model=ChatCompletionResponse)
async def agent_team_chat(
    request: ChatCompletionRequest,
    team: AgentTeamConfig,
    task_type: TaskType = "chat",
    team_selection: TeamSelection = "default"
):
    """Handle agent team chat requests."""
    # Route to first agent in team for now
    # TODO: Implement proper team routing
    return await route_to_agent(request, team.agents[0], task_type, team_selection)
