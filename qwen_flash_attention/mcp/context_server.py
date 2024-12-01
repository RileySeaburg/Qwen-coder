from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import os
import json
import time
import httpx
from .tools import (
    ToolRegistry, ToolDefinition, ToolRequest, ToolResponse,
    default_registry
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Context Protocol Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Model Context Protocol types
class MCPMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class MCPRequest(BaseModel):
    messages: List[MCPMessage]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2048, gt=0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[str]] = None

class MCPChoice(BaseModel):
    index: int
    message: MCPMessage
    finish_reason: str

class MCPUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class MCPResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[MCPChoice]
    usage: Optional[MCPUsage] = None

class MCPError(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: str

@app.on_event("startup")
async def startup_event():
    """Initialize tool registry on startup."""
    try:
        await default_registry.initialize()
        logger.info("Tool registry initialized")
    except Exception as e:
        logger.error(f"Error initializing tool registry: {e}")
        raise

@app.get("/v1/tools")
async def list_tools() -> Dict[str, List[ToolDefinition]]:
    """List available tools."""
    return {
        "data": default_registry.list_tools()
    }

@app.post("/v1/tools/execute")
async def execute_tool(request: ToolRequest) -> ToolResponse:
    """Execute a tool."""
    return await default_registry.execute_tool(request)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: MCPRequest) -> MCPResponse:
    """Create a chat completion."""
    try:
        # Get available tools if requested
        available_tools = []
        if request.tools:
            available_tools = [
                tool for tool in default_registry.list_tools()
                if tool.name in request.tools
            ]

        # Format messages for model
        formatted_messages = []
        for msg in request.messages:
            if msg.role == "tool":
                # Format tool response
                formatted_messages.append({
                    "role": "tool",
                    "name": msg.name,
                    "content": msg.content
                })
            elif msg.tool_calls:
                # Format tool calls
                tool_calls_text = ""
                for tool_call in msg.tool_calls:
                    tool_calls_text += f"Using tool {tool_call['name']}\n"
                    tool_calls_text += f"```tool\n{json.dumps(tool_call, indent=2)}\n```\n"
                formatted_messages.append({
                    "role": "assistant",
                    "content": tool_calls_text
                })
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Forward request to model server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/generate",
                json={
                    "messages": formatted_messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "tools": [tool.dict() for tool in available_tools] if available_tools else None
                }
            )
            response.raise_for_status()
            model_response = response.json()

        # Parse tool calls if present
        tool_calls = []
        if "```tool" in model_response["text"]:
            # Extract tool calls from markdown code blocks
            parts = model_response["text"].split("```tool")
            for part in parts[1:]:
                try:
                    tool_json = part.split("```")[0].strip()
                    tool_call = json.loads(tool_json)
                    tool_calls.append(tool_call)
                except:
                    continue

        return MCPResponse(
            id=f"chatcmpl-{hash(model_response['text'])}",
            created=int(time.time()),
            model="qwen2.5-coder-1.5b",
            choices=[
                MCPChoice(
                    index=0,
                    message=MCPMessage(
                        role="assistant",
                        content=model_response["text"],
                        tool_calls=tool_calls if tool_calls else None
                    ),
                    finish_reason="stop"
                )
            ],
            usage=MCPUsage(
                prompt_tokens=model_response["usage"]["prompt_tokens"],
                completion_tokens=model_response["usage"]["completion_tokens"],
                total_tokens=model_response["usage"]["total_tokens"]
            )
        )
    except Exception as e:
        logger.error(f"Error creating chat completion: {e}")
        raise HTTPException(
            status_code=500,
            detail=MCPError(
                message=str(e),
                type="internal_error",
                code="model_error"
            ).dict()
        )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=MCPError(
            message=str(exc),
            type="internal_error",
            code="server_error"
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
