from typing import List, Dict, Any

class ToolParameter:
    def __init__(self, name: str, type: str, description: str, required: bool = True):
        self.name = name
        self.type = type
        self.description = description
        self.required = required

    def dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required
        }

class ToolDefinition:
    def __init__(self, name: str, description: str, parameters: List[ToolParameter], returns: Dict[str, str]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns = returns

    def dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.dict() for p in self.parameters],
            "returns": self.returns
        }

# Define browser action tool
browser_action = ToolDefinition(
    name="browser_action",
    description="Control a browser to visit websites and interact with web pages",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="The action to perform (launch, click, type, scroll_down, scroll_up, close)",
            required=True
        ),
        ToolParameter(
            name="url",
            type="string",
            description="URL to visit (required for launch action)",
            required=False
        ),
        ToolParameter(
            name="coordinate",
            type="string",
            description="x,y coordinates for click action",
            required=False
        ),
        ToolParameter(
            name="text",
            type="string",
            description="Text to type",
            required=False
        )
    ],
    returns={
        "type": "object",
        "description": "Browser action result including screenshot and logs"
    }
)

# Define file operation tools
read_file = ToolDefinition(
    name="readFile",
    description="Read contents of a file",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file",
            required=True
        )
    ],
    returns={
        "type": "string",
        "description": "File contents"
    }
)

write_file = ToolDefinition(
    name="writeFile",
    description="Write content to a file",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file",
            required=True
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write",
            required=True
        )
    ],
    returns={
        "type": "boolean",
        "description": "Success status"
    }
)

# Map of available tools
available_tools = {
    "browser_action": browser_action,
    "readFile": read_file,
    "writeFile": write_file
}

def get_tool_definition(tool_name: str) -> Dict[str, Any]:
    """Get the definition for a specific tool."""
    if tool_name not in available_tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    return available_tools[tool_name].dict()

def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """Get OpenAPI-compatible schema for a tool."""
    tool_def = get_tool_definition(tool_name)
    return {
        "type": "object",
        "properties": {
            param["name"]: {
                "type": param["type"],
                "description": param["description"]
            }
            for param in tool_def["parameters"]
        },
        "required": [
            param["name"]
            for param in tool_def["parameters"]
            if param["required"]
        ]
    }
