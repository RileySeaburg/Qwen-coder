import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, cast
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .model import QwenModel
from .tool_model import ToolModel
from .websocket_manager import WebSocketManager
from .rag_agent import RAGAgent
from .chat_routes import router as chat_router, init_models
from .autogen_router import AutoGenRouter
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WebSocket manager
ws_manager = WebSocketManager()
logger.info("WebSocket manager initialized")

# Initialize AutoGen router
autogen_router = AutoGenRouter()

@app.on_event("startup")
async def startup():
    """Initialize models on startup."""
    try:
        # Initialize with Qwen 32B model
        logger.info("Initializing Qwen model...")
        qwen_model = QwenModel("Qwen/Qwen2.5-32B")
        logger.info("Qwen model initialized successfully")
        
        logger.info("Initializing ToolACE model...")
        tool_model = ToolModel()
        logger.info("ToolACE model initialized successfully")
        
        logger.info("Initializing RAG agent...")
        rag_agent = RAGAgent()
        logger.info("RAG agent initialized successfully")
        
        # Initialize models in chat routes
        init_models(qwen_model, tool_model, rag_agent)
        
        # Initialize AutoGen router
        logger.info("Initializing AutoGen router...")
        await autogen_router.initialize()
        logger.info("AutoGen router initialized successfully")
        
    except Exception as e:
        logger.error(f"Error in startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown."""
    logger.info("Cleaning up resources...")
    # Clear model instances without calling init_models
    logger.info("Cleanup completed")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "chat":
                # Convert to ChatRequest
                request = ChatCompletionRequest(**message["data"])
                
                # Get response
                try:
                    # Call chat completion endpoint directly
                    response = await app.dependency_overrides[chat_router.url_path_for("chat_completion_endpoint").name](request)
                    await websocket.send_json(response.dict())
                except Exception as e:
                    logger.error(f"Error in websocket chat: {e}")
                    await websocket.send_json({
                        "error": str(e),
                        "status_code": 500
                    })
                
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/v1/models")
async def list_models() -> Dict[str, List[Dict[str, Optional[Union[str, List[Any]]]]]]:
    """List available models."""
    return {
        "data": [
            {
                "id": "Qwen/Qwen2.5-32B",
                "object": "model",
                "created": "1677858242",
                "owned_by": "Qwen",
                "permission": [],
                "root": None,
                "parent": None
            }
        ]
    }

# Include chat routes
app.include_router(chat_router)

# Add AutoGen routes
@app.post("/v1/autogen/chat/completions", response_model=ChatCompletionResponse)
async def autogen_chat_completion(request: ChatCompletionRequest, agent: Agent):
    """Handle AutoGen chat completion requests."""
    try:
        if agent.model.startswith("claude"):
            return await autogen_router.route_to_claude(request, agent)
        else:
            return await autogen_router.route_to_qwen(request, agent)
    except Exception as e:
        logger.error(f"Error in AutoGen chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/autogen/team/chat", response_model=ChatCompletionResponse)
async def autogen_team_chat(
    request: ChatCompletionRequest,
    team: AgentTeamConfig,
    task_type: TaskType = "chat",
    team_selection: TeamSelection = "default"
):
    """Handle AutoGen team chat requests."""
    try:
        # Route to first agent in team for now
        # TODO: Implement proper team routing
        agent = team.agents[0]
        if agent.model.startswith("claude"):
            return await autogen_router.route_to_claude(request, agent)
        else:
            return await autogen_router.route_to_qwen(request, agent)
    except Exception as e:
        logger.error(f"Error in AutoGen team chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
