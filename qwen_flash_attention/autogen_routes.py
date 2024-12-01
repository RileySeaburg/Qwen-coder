import logging
import traceback
import re
import os
import time
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List, Optional
from typing_extensions import NotRequired
from pydantic import BaseModel, Field
from .schemas import Message, Agent, AgentTeamConfig
from .rag_autogen import RAGAutogenTeam
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any]

class KnowledgeAddRequest(BaseModel):
    documents: List[Document]
    source: str

class AgentTeam(BaseModel):
    agents: List[Agent] = Field(..., description="List of agents in the team")
    task: str = Field(..., description="Task to be solved")

class MessageDict(BaseModel):
    role: str
    content: str
    name: str

class TeamSolveChoice(BaseModel):
    index: int
    message: MessageDict
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class TeamSolveResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[TeamSolveChoice]
    usage: Optional[UsageInfo] = None

# Store active teams
active_teams: Dict[str, RAGAutogenTeam] = {}

def determine_task_type(message: str) -> str:
    """Determine the type of task from the message."""
    message = message.lower()
    
    patterns = {
        "code": r"(code|program|function|class|implement|debug|fix|refactor)",
        "architecture": r"(design|architect|structure|pattern|system)",
        "review": r"(review|analyze|assess|evaluate|check)",
        "test": r"(test|unit test|integration test|qa|quality)",
        "documentation": r"(document|comment|explain|describe)",
        "devops": r"(deploy|ci/cd|pipeline|docker|kubernetes)",
        "database": r"(database|sql|query|schema|model)",
        "security": r"(security|auth|encrypt|protect|vulnerability)"
    }
    
    matches = {task: bool(re.search(pattern, message)) 
              for task, pattern in patterns.items()}
    
    # Return the task type with the most matches, or "general" if no matches
    matched_tasks = [task for task, matched in matches.items() if matched]
    return matched_tasks[0] if matched_tasks else "general"

async def route_to_qwen(request: AgentTeam, agent: Agent, qwen_model: Any) -> MessageDict:
    """Route request to local Qwen API."""
    try:
        logger.info(f"Routing to Qwen API for agent {agent.name}")

        # Format task as message
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": agent.systemPrompt,
                "name": agent.name
            },
            {
                "role": "user",
                "content": request.task,
                "name": "user"
            }
        ]

        # Generate response
        response = await qwen_model.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        return MessageDict(
            role="assistant",
            content=response,
            name=agent.name
        )
    except Exception as e:
        logger.error(f"Error routing to Qwen: {e}")
        raise

@router.post("/v1/autogen/knowledge/add")
async def add_knowledge(request: KnowledgeAddRequest, fastapi_request: Request):
    """Add knowledge to the vector database."""
    try:
        logger.info(f"Adding knowledge from source: {request.source}")
        logger.info(f"Number of documents: {len(request.documents)}")
        
        # Get RAG agent from app state
        rag_agent = fastapi_request.app.state.rag_agent
        if not rag_agent:
            logger.error("RAG agent not initialized")
            raise HTTPException(status_code=500, detail="RAG agent not initialized")

        # Get embeddings directly from Qwen model
        qwen_model = fastapi_request.app.state.qwen_model
        if not qwen_model:
            logger.error("Qwen model not initialized")
            raise HTTPException(status_code=500, detail="Qwen model not initialized")

        # Get embeddings for all documents
        texts = [doc.text for doc in request.documents]
        logger.info(f"Getting embeddings for {len(texts)} documents")
        try:
            embeddings = qwen_model.get_embeddings(texts)
            logger.info(f"Got embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

        # Convert numpy array to list
        try:
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            logger.info(f"Converted embeddings to list format")
        except Exception as e:
            logger.error(f"Error converting embeddings to list: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error converting embeddings: {str(e)}")

        # Create MongoDB documents
        try:
            mongo_docs = []
            for doc, embedding in zip(request.documents, embeddings_list):
                mongo_doc = {
                    "text": doc.text,
                    "embedding": embedding,
                    "metadata": doc.metadata
                }
                mongo_docs.append(mongo_doc)
            logger.info(f"Created {len(mongo_docs)} MongoDB documents")
            logger.debug(f"First document sample: {mongo_docs[0] if mongo_docs else 'No documents'}")
        except Exception as e:
            logger.error(f"Error creating MongoDB documents: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error creating documents: {str(e)}")

        # Add documents to vector store
        try:
            doc_ids = await rag_agent.add_documents(mongo_docs, request.source)
            logger.info(f"Added documents to vector store, got {len(doc_ids)} IDs")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error storing documents: {str(e)}")

        return {
            "status": "success",
            "message": f"Added {len(doc_ids)} documents",
            "document_ids": [str(doc_id) for doc_id in doc_ids]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in add_knowledge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/v1/autogen/team/solve")
async def team_solve(request: AgentTeam, fastapi_request: Request) -> TeamSolveResponse:
    """Solve a task using a team of agents."""
    try:
        logger.debug(f"Starting team solve with {len(request.agents)} agents")
        logger.debug(f"Task: {request.task}")
        
        # Get model instance from app state
        qwen_model = fastapi_request.app.state.qwen_model
        
        # Create team if not exists
        team_id = f"team_{len(active_teams)}"
        if team_id not in active_teams:
            # Create config list for models
            config_list = []
            for agent in request.agents:
                if "claude" in agent.model.lower():
                    config_list.append({
                        "model": agent.model,
                        "api_key": os.getenv("CLAUDE_API_KEY"),
                        "api_type": "anthropic"
                    })
                else:
                    config_list.append({
                        "model": agent.model,
                        "api_type": "openai",
                        "base_url": "http://localhost:8000/v1"
                    })

            # Create and initialize team
            team = RAGAutogenTeam(
                config_list=config_list,
                mongodb_config={
                    "url": "mongodb://localhost:27017",
                    "database": "autogen_vectors",
                    "collection": "team_knowledge",
                    "model_name": "Qwen/Qwen2.5-Coder-3B"
                },
                request=fastapi_request
            )
            await team.initialize()
            active_teams[team_id] = team

        # Get team
        team = active_teams[team_id]

        # Process task
        response = await team.process_task(request.task)

        # Format response
        message = MessageDict(
            role="assistant",
            content=response["content"],
            name=response.get("name", "team")
        )

        choice = TeamSolveChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )

        team_response = TeamSolveResponse(
            id=f"solve_{team_id}",
            object="team.solve",
            created=int(time.time()),
            model="team",
            choices=[choice],
            usage=None
        )

        return team_response

    except Exception as e:
        logger.error(f"Error in team solve: {e}")
        raise HTTPException(status_code=500, detail=str(e))
