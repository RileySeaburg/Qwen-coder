from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import logging
from .code_model import QwenForCausalLM
from .document_manager import DocumentManager, AddDocumentsRequest, AddDocumentsResponse

logger = logging.getLogger("api_server")

router = APIRouter()

# Global variables shared with api_server.py
available_models: Dict[str, Dict[str, Any]] = {}
USE_FLASH_ATTENTION: bool = False

def init_globals(models: Dict[str, Dict[str, Any]], flash_attention: bool) -> None:
    """Initialize global variables."""
    global available_models, USE_FLASH_ATTENTION
    available_models = models
    USE_FLASH_ATTENTION = flash_attention
    logger.info(f"Model routes initialized with Flash Attention {'enabled' if flash_attention else 'disabled'}")

def get_model_instance(model_name: str) -> QwenForCausalLM:
    """Get model instance."""
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
    
    # QwenForCausalLM is a singleton, so we just return the instance
    return QwenForCausalLM()

@router.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """List available models."""
    return {
        "data": [
            {
                "id": model["name"],
                "object": "model",
                "created": None,
                "owned_by": "Qwen",
                "permission": [],
                "root": model["repo_id"],
                "parent": None,
                "capabilities": {
                    "chat_completion": True,
                    "embeddings": True
                }
            }
            for model in available_models.values()
        ],
        "object": "list"
    }

@router.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    if model_id not in available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model = available_models[model_id]
    return {
        "id": model["name"],
        "object": "model",
        "created": None,
        "owned_by": "Qwen",
        "permission": [],
        "root": model["repo_id"],
        "parent": None,
        "capabilities": {
            "chat_completion": True,
            "embeddings": True
        }
    }

@router.post("/v1/documents/add")
async def add_documents(request: AddDocumentsRequest, fastapi_request: Request) -> AddDocumentsResponse:
    """Add documents to the knowledge base with embeddings."""
    doc_manager = DocumentManager(fastapi_request)
    return await doc_manager.add_documents(request)
