from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Callable, Sequence
from enum import Enum
import logging
import os
import subprocess
import json
from ..rag_agent import RAGAgent, SearchResult, MongoDocument

logger = logging.getLogger(__name__)

class ToolParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"

class ToolParameter(BaseModel):
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    items: Optional[Dict[str, str]] = None
    properties: Optional[Dict[str, Any]] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: Dict[str, str]

class ToolRequest(BaseModel):
    tool: str
    parameters: Dict[str, Any]

class ToolResponse(BaseModel):
    result: Any
    error: Optional[str] = None

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, tuple[ToolDefinition, Callable]] = {}
        self.rag_agent = RAGAgent(
            mongodb_url="mongodb://localhost:27017",
            database_name="autogen_vectors",
            collection_name="team_knowledge"
        )

    async def initialize(self):
        """Initialize RAG agent."""
        await self.rag_agent.initialize()
        logger.info("Tool registry initialized with RAG support")

    def register_tool(self, definition: ToolDefinition, handler: Callable):
        """Register a tool with its handler."""
        self.tools[definition.name] = (definition, handler)
        logger.info(f"Registered tool: {definition.name}")

    async def execute_tool(self, request: ToolRequest) -> ToolResponse:
        """Execute a tool with given parameters."""
        if request.tool not in self.tools:
            return ToolResponse(result=None, error=f"Tool '{request.tool}' not found")

        definition, handler = self.tools[request.tool]

        try:
            # Validate parameters
            for param in definition.parameters:
                if param.required and param.name not in request.parameters:
                    raise ValueError(f"Missing required parameter: {param.name}")

            # Execute tool
            result = await handler(**request.parameters)
            return ToolResponse(result=result)
        except Exception as e:
            logger.error(f"Error executing tool {request.tool}: {e}")
            return ToolResponse(result=None, error=str(e))

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return [definition for definition, _ in self.tools.values()]

# Built-in tools
async def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

async def write_file(path: str, content: str) -> bool:
    """Write contents to a file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        raise ValueError(f"Failed to write file: {e}")

async def execute_command(command: str) -> Dict[str, Any]:
    """Execute a shell command."""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": process.returncode
        }
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}")

async def search_knowledge(query: str, num_results: int = 3) -> Sequence[Dict[str, Any]]:
    """Search knowledge base."""
    try:
        results: List[SearchResult] = await default_registry.rag_agent.search_similar(
            query=query,
            num_results=num_results
        )
        # Convert SearchResult to dict
        return [dict(result) for result in results]
    except Exception as e:
        raise ValueError(f"Failed to search knowledge base: {e}")

async def add_knowledge(text: str, metadata: Dict[str, Any]) -> bool:
    """Add knowledge to vector store."""
    try:
        document: MongoDocument = {
            "text": text,
            "embedding": [],  # Will be filled by RAG agent
            "metadata": metadata
        }
        result = await default_registry.rag_agent.add_documents(
            documents=[document],
            source=metadata.get("source", "tool_input")
        )
        return bool(result)
    except Exception as e:
        raise ValueError(f"Failed to add knowledge: {e}")

# Create default registry with built-in tools
default_registry = ToolRegistry()

# Register built-in tools
default_registry.register_tool(
    ToolDefinition(
        name="readFile",
        description="Read contents of a file",
        parameters=[
            ToolParameter(
                name="path",
                type=ToolParameterType.STRING,
                description="Path to the file"
            )
        ],
        returns={"type": "string", "description": "File contents"}
    ),
    read_file
)

default_registry.register_tool(
    ToolDefinition(
        name="writeFile",
        description="Write contents to a file",
        parameters=[
            ToolParameter(
                name="path",
                type=ToolParameterType.STRING,
                description="Path to the file"
            ),
            ToolParameter(
                name="content",
                type=ToolParameterType.STRING,
                description="Content to write"
            )
        ],
        returns={"type": "boolean", "description": "Success status"}
    ),
    write_file
)

default_registry.register_tool(
    ToolDefinition(
        name="executeCommand",
        description="Execute a shell command",
        parameters=[
            ToolParameter(
                name="command",
                type=ToolParameterType.STRING,
                description="Command to execute"
            )
        ],
        returns={"type": "object", "description": "Command execution result"}
    ),
    execute_command
)

default_registry.register_tool(
    ToolDefinition(
        name="searchKnowledge",
        description="Search knowledge base",
        parameters=[
            ToolParameter(
                name="query",
                type=ToolParameterType.STRING,
                description="Search query"
            ),
            ToolParameter(
                name="num_results",
                type=ToolParameterType.NUMBER,
                description="Number of results to return",
                required=False
            )
        ],
        returns={"type": "array", "description": "Search results"}
    ),
    search_knowledge
)

default_registry.register_tool(
    ToolDefinition(
        name="addKnowledge",
        description="Add knowledge to vector store",
        parameters=[
            ToolParameter(
                name="text",
                type=ToolParameterType.STRING,
                description="Text content"
            ),
            ToolParameter(
                name="metadata",
                type=ToolParameterType.OBJECT,
                description="Metadata for the document"
            )
        ],
        returns={"type": "boolean", "description": "Success status"}
    ),
    add_knowledge
)
