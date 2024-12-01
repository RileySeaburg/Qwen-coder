from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import json
from mcp.tools import (
    ToolRegistry, ToolDefinition, ToolRequest, ToolResponse,
    default_registry
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CUDA memory management
torch.cuda.set_per_process_memory_fraction(0.8)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

app = FastAPI(title="Qwen Model Context Protocol Server")

TOOL_SYSTEM_PROMPT = """You are an AI assistant with access to tools. When you need to use a tool:
1. Format your tool call in a markdown code block with the language 'tool'
2. Include the tool name and parameters as JSON
3. Wait for the tool response before proceeding

Example tool usage:
```tool
{
    "name": "readFile",
    "parameters": {
        "path": "/path/to/file"
    }
}
```

Available tools:
{tool_list}

Remember to:
- Only use available tools
- Format tool calls exactly as shown
- Handle tool responses appropriately
- One tool call per code block"""

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

class MCPModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[str]
    root: Optional[str] = None
    parent: Optional[str] = None
    context_length: int = 2048

class MCPError(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: str

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

    def format_tool_list(self, tools: List[ToolDefinition]) -> str:
        """Format tool list for system prompt."""
        tool_docs = []
        for tool in tools:
            params = [
                f"- {p.name}: {p.type} - {p.description} ({'required' if p.required else 'optional'})"
                for p in tool.parameters
            ]
            tool_docs.append(
                f"{tool.name}: {tool.description}\n"
                f"Parameters:\n"
                f"{chr(10).join(params)}\n"
                f"Returns: {tool.returns['type']} - {tool.returns['description']}"
            )
        return "\n\n".join(tool_docs)

    async def generate(self, messages: List[MCPMessage], temperature: float, max_tokens: int, tools: List[ToolDefinition] = []) -> str:
        # Format messages into prompt
        prompt = ""
        
        # Add tool system prompt if tools are available
        if tools:
            tool_list = self.format_tool_list(tools)
            prompt += TOOL_SYSTEM_PROMPT.format(tool_list=tool_list) + "\n\n"

        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Format tool calls
                    for tool_call in msg.tool_calls:
                        prompt += f"Assistant: Using tool {tool_call['name']}\n"
                        prompt += f"```tool\n{json.dumps(tool_call, indent=2)}\n```\n"
                else:
                    prompt += f"Assistant: {msg.content}\n"
            elif msg.role == "tool":
                prompt += f"Tool ({msg.name}): {msg.content}\n"

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = response[len(prompt):].strip()
        return response

# Initialize model and tool registry
model = QwenModel()
tool_registry = default_registry

@app.on_event("startup")
async def startup_event():
    """Initialize tool registry on startup."""
    try:
        await tool_registry.initialize()
        logger.info("Tool registry initialized")
    except Exception as e:
        logger.error(f"Error initializing tool registry: {e}")
        raise

@app.get("/v1/models")
async def list_models() -> Dict[str, List[MCPModelCard]]:
    """List available models."""
    return {
        "data": [
            MCPModelCard(
                id="Qwen2.5-Coder-1.5B-Instruct",
                created=1700000000,
                owned_by="Qwen",
                permission=["read", "write"],
                context_length=2048
            )
        ]
    }

@app.get("/v1/tools")
async def list_tools() -> Dict[str, List[ToolDefinition]]:
    """List available tools."""
    return {
        "data": tool_registry.list_tools()
    }

@app.post("/v1/tools/execute")
async def execute_tool(request: ToolRequest) -> ToolResponse:
    """Execute a tool."""
    try:
        response = await tool_registry.execute_tool(request)
        if response.error:
            raise HTTPException(
                status_code=400,
                detail=MCPError(
                    message=response.error,
                    type="tool_error",
                    code="tool_execution_failed"
                ).dict()
            )
        return response
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(
            status_code=500,
            detail=MCPError(
                message=str(e),
                type="internal_error",
                code="tool_error"
            ).dict()
        )

@app.post("/v1/chat/completions")
async def create_chat_completion(request: MCPRequest) -> MCPResponse:
    """Create a chat completion."""
    try:
        # Get available tools if requested
        available_tools = []
        if request.tools:
            available_tools = [
                tool for tool in tool_registry.list_tools()
                if tool.name in request.tools
            ]

        # Generate response
        response_text = await model.generate(
            messages=request.messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2048,
            tools=available_tools
        )

        # Parse tool calls if present
        tool_calls = []
        if "```tool" in response_text:
            # Extract tool calls from markdown code blocks
            parts = response_text.split("```tool")
            for part in parts[1:]:
                try:
                    tool_json = part.split("```")[0].strip()
                    tool_call = json.loads(tool_json)
                    tool_calls.append(tool_call)
                except:
                    continue

        # Count tokens
        input_text = "\n".join(msg.content for msg in request.messages)
        input_tokens = len(model.tokenizer.encode(input_text))
        output_tokens = len(model.tokenizer.encode(response_text))

        return MCPResponse(
            id=f"chatcmpl-{hash(response_text)}",
            created=int(torch.cuda.initial_seed()),
            model=model.model_name,
            choices=[
                MCPChoice(
                    index=0,
                    message=MCPMessage(
                        role="assistant",
                        content=response_text,
                        tool_calls=tool_calls if tool_calls else None
                    ),
                    finish_reason="stop"
                )
            ],
            usage=MCPUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
        )
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
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
