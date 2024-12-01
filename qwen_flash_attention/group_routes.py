import logging
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List, Optional, cast, Sequence, Literal
from pydantic import BaseModel
from .schemas import AgentTeamConfig, Message, ChatCompletionRequest, Agent
from .chat_routes import ChatResponse, chat_completion
import time

logger = logging.getLogger(__name__)

router = APIRouter()
active_groups: Dict[str, AgentTeamConfig] = {}

class CreateGroupRequest(BaseModel):
    agents: List[Agent]
    teamType: Literal["round_robin", "hierarchical"]

class GroupResponse(BaseModel):
    group_id: str
    object: str = "group.create"
    created: int
    agents: Sequence[Agent]
    teamType: Literal["round_robin", "hierarchical"]

class GroupChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    tools: Optional[List[str]] = None

class GroupChatResponse(BaseModel):
    id: str
    object: str = "group.chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

@router.post("/v1/groups/create")
async def create_group(request: CreateGroupRequest) -> GroupResponse:
    """Create a new agent group."""
    try:
        # Generate group ID
        group_id = f"group_{len(active_groups)}"
        created = int(time.time())

        # Store group configuration
        config = AgentTeamConfig(
            agents=request.agents,
            teamType=request.teamType
        )
        active_groups[group_id] = config

        return GroupResponse(
            group_id=group_id,
            created=created,
            agents=request.agents,
            teamType=request.teamType
        )
    except Exception as e:
        logger.error(f"Error creating group: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/groups/{group_id}/chat")
async def group_chat(group_id: str, request: GroupChatRequest, fastapi_request: Request) -> GroupChatResponse:
    """Chat with a specific agent group."""
    try:
        if group_id not in active_groups:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

        # Get group config
        config = active_groups[group_id]
        logger.info(f"Using group config: {config}")

        # Add browser_action tool if web_researcher role is present
        tools = request.tools or []
        if any(agent.role == "web_researcher" for agent in config.agents):
            if "browser_action" not in tools:
                tools.append("browser_action")
            logger.info("Added browser_action tool for web researcher")

        # Create chat completion request
        chat_request = ChatCompletionRequest(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            agent_config=config,
            tools=tools
        )
        
        # Get response from chat completion
        chat_response = await chat_completion(chat_request, fastapi_request)
        logger.info(f"Got chat response: {chat_response}")

        # Convert chat choices to dict format
        choices = [
            {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                    "name": choice.message.name
                },
                "finish_reason": choice.finish_reason
            }
            for choice in chat_response.choices
        ]

        # Format as group chat response
        return GroupChatResponse(
            id=f"groupchat-{group_id}-{int(time.time())}",
            created=int(time.time()),
            model=config.agents[0].model if config.agents else "unknown",
            choices=choices,
            usage=chat_response.usage.dict() if chat_response.usage else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in group chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/v1/groups/{group_id}")
async def delete_group(group_id: str):
    """Delete an agent group."""
    try:
        if group_id not in active_groups:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        
        del active_groups[group_id]
        return {"status": "success", "message": f"Group {group_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting group: {e}")
        raise HTTPException(status_code=500, detail=str(e))
