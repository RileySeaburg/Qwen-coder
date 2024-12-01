from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, constr

class Message(BaseModel):
    role: str
    content: str = Field(max_length=32768)  # 32KB limit for content
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class AgentId(BaseModel):
    type: str
    key: str

class AgentState(BaseModel):
    agent_id: AgentId
    eTag: str
    binary_data: Optional[bytes] = None
    text_data: Optional[str] = None

class Agent(BaseModel):
    id: Optional[AgentId] = None
    name: str
    role: str
    model: str
    systemPrompt: str = Field(max_length=32768)  # 32KB limit for system prompts

class AgentTeamConfig(BaseModel):
    agents: List[Agent]
    teamType: Literal["round_robin", "hierarchical"]

TaskType = Literal["chat", "solve"]
TeamSelection = Literal["default", "custom"]

class EnvironmentDetails(BaseModel):
    visible_files: List[str] = Field(default_factory=list)
    open_tabs: List[str] = Field(default_factory=list)
    current_directory: str
    files: List[str] = Field(default_factory=list)

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=0.7, gt=0, description="Temperature for sampling. Must be greater than 0.")
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    tools: Optional[List[str]] = None
    agent_config: Optional[AgentTeamConfig] = None
    task_type: Optional[TaskType] = Field(default="chat", description="Type of task: chat or solve")
    team_selection: Optional[TeamSelection] = Field(default="default", description="Team selection: default or custom")
    environment_details: Optional[EnvironmentDetails] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class BrowserActionMetadata(BaseModel):
    screenshot: Optional[str] = None
    logs: Optional[List[str]] = None
    currentUrl: Optional[str] = None
    currentMousePosition: Optional[str] = None

class AgentMetadata(BaseModel):
    agent: Dict[str, str]

class MessageMetadata(BaseModel):
    agent: Optional[Dict[str, str]] = None
    browserAction: Optional[BrowserActionMetadata] = None

class BaseQwenMessageWithMetadata(BaseModel):
    ts: int
    partial: Optional[bool] = None
    images: Optional[List[str]] = None
    text: Optional[str] = None
    role: Optional[str] = None
    metadata: Optional[MessageMetadata] = None

class QwenMessageParam(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class QwenAskResponse(BaseModel):
    type: str = "ask_response"
    response: str
    text: Optional[str] = None
    images: Optional[List[str]] = None
    role: Optional[str] = None

class ExtensionState(BaseModel):
    version: str
    apiConfiguration: Dict[str, Any]
    customInstructions: str
    alwaysAllowReadOnly: bool
    messages: List[Dict[str, Any]]
    taskHistory: List[Any]
    shouldShowAnnouncement: bool
    qwenMessages: List[Dict[str, Any]]
    agentTeamConfig: Optional[AgentTeamConfig] = None
    taskType: Optional[TaskType] = None
    teamSelection: Optional[TeamSelection] = None

class QwenState(BaseModel):
    messages: List[Dict[str, Any]]
    apiConfiguration: Dict[str, Any]
    customInstructions: str
    alwaysAllowReadOnly: bool
    taskHistory: List[Any]
    shouldShowAnnouncement: bool
    agentTeamConfig: Optional[AgentTeamConfig] = None
    taskType: Optional[TaskType] = None
    teamSelection: Optional[TeamSelection] = None
