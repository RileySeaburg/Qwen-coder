from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the message sender")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = {'user', 'assistant', 'system'}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class Agent(BaseModel):
    name: str = Field(..., description="The name of the agent")
    role: str = Field(..., description="The role of the agent (e.g., 'assistant', 'user')")
    model: str = Field(..., description="The model to use for this agent")
    systemPrompt: str = Field(..., description="The system prompt for this agent")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = {'assistant', 'user'}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class AgentTeamConfig(BaseModel):
    agents: List[Agent] = Field(..., description="List of agents in the team")
    teamType: str = Field(
        "round_robin",
        description="Type of team interaction ('round_robin' or 'hierarchical')"
    )

    @field_validator('teamType')
    @classmethod
    def validate_team_type(cls, v):
        allowed_types = {'round_robin', 'hierarchical'}
        if v not in allowed_types:
            raise ValueError(f"Team type must be one of {allowed_types}")
        return v

    @field_validator('agents')
    @classmethod
    def validate_agents(cls, v):
        if not v:
            raise ValueError("At least one agent must be specified")
        return v

class ChatCompletionRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(None, description="Optional model override")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.95, description="Top-p sampling parameter")
    max_tokens: Optional[int] = Field(2048, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    agent_config: Optional[AgentTeamConfig] = Field(
        None,
        description="Optional agent team configuration for multi-agent interactions"
    )
    use_memory: Optional[bool] = Field(True, description="Whether to use memory for context")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("Top-p must be between 0 and 1")
        return v

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Max tokens must be greater than 0")
        return v

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message must be provided")
        return v

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the response")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="List of completion choices")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")

class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Type of websocket message")
    agent: Dict[str, str] = Field(..., description="Agent information")
    content: str = Field(..., description="Message content")
    timestamp: float = Field(..., description="Unix timestamp of message")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        allowed_types = {'agent_message', 'system_message', 'error_message'}
        if v not in allowed_types:
            raise ValueError(f"Message type must be one of {allowed_types}")
        return v

    @field_validator('agent')
    @classmethod
    def validate_agent(cls, v):
        required_fields = {'name', 'role'}
        if not all(field in v for field in required_fields):
            raise ValueError(f"Agent must contain fields: {required_fields}")
        return v
