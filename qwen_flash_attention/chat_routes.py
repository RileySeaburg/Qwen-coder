import logging
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, APIRouter, Depends
from pydantic import BaseModel

from .model import QwenModel
from .tool_model import ToolModel
from .rag_agent import RAGAgent

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "Team-ACE/ToolACE-8B"
    temperature: float = 0.7
    max_tokens: int = 1024
    tools: Optional[List[str]] = None
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# Global model instances
_qwen_model: Optional[QwenModel] = None
_tool_model: Optional[ToolModel] = None
_rag_agent: Optional[RAGAgent] = None

def get_qwen_model() -> QwenModel:
    if _qwen_model is None:
        raise HTTPException(status_code=500, detail="Qwen model not initialized")
    return _qwen_model

def get_tool_model() -> ToolModel:
    if _tool_model is None:
        raise HTTPException(status_code=500, detail="Tool model not initialized")
    return _tool_model

def get_rag_agent() -> RAGAgent:
    if _rag_agent is None:
        raise HTTPException(status_code=500, detail="RAG agent not initialized")
    return _rag_agent

def init_models(qwen: QwenModel, tool: ToolModel, rag: RAGAgent):
    """Initialize global model instances."""
    global _qwen_model, _tool_model, _rag_agent
    _qwen_model = qwen
    _tool_model = tool
    _rag_agent = rag

async def chat_completion(
    request: ChatRequest,
    qwen_model: QwenModel = Depends(get_qwen_model),
    tool_model: ToolModel = Depends(get_tool_model),
    rag_agent: RAGAgent = Depends(get_rag_agent)
) -> ChatResponse:
    """Handle chat completion requests."""
    try:
        logger.info(f"Received chat completion request with {len(request.messages)} messages")
        
        # Determine task type and team selection
        task_type = "chat"
        team_selection = "default"
        logger.info(f"Task type: {task_type}, Team selection: {team_selection}")
        
        # Convert messages to list of dicts
        messages = [msg.dict(exclude_none=True) for msg in request.messages]
        logger.info(f"Processing messages: {messages}")
        
        # Use tool model for tool interactions
        if request.tools:
            logger.info("Using ToolACE model for tool interactions")
            response = await tool_model.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools
            )
            
            # Convert function call format if needed
            if "[" in response and "(" in response:
                # Extract function name and arguments
                import re
                match = re.match(r'\[(.*?)\((.*?)\)\]', response)
                if match:
                    func_name = match.group(1)
                    args_str = match.group(2)
                    
                    # Parse arguments
                    args = {}
                    for arg in args_str.split(','):
                        key, value = arg.split('=')
                        args[key.strip()] = value.strip().strip("'\"")
                    
                    # Convert to JSON format
                    response = f"```json\n{{\n  \"name\": \"{func_name}\",\n  \"arguments\": {{\n"
                    for key, value in args.items():
                        response += f"    \"{key}\": \"{value}\",\n"
                    response = response.rstrip(",\n") + "\n  }\n}}\n```"
            
        else:
            # Use RAG agent for knowledge-based responses
            logger.info("Using RAG agent for knowledge-based response")
            response = await rag_agent.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
        return ChatResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677858242,
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
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
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add routes
@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle chat completion requests."""
    return await chat_completion(request)
