from typing import List, Dict, Optional, Union
import numpy as np
from .vector_store import VectorStore, VectorDocument
from .config import settings
import logging
import json

logger = logging.getLogger("vector_store")
logger.setLevel(logging.INFO)

class VectorStoreManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance.store = VectorStore(
                mongodb_url=settings.MONGODB_URL,
                database_name=settings.DATABASE_NAME,
                collection_name=settings.COLLECTION_NAME,
                model_name=settings.MODEL_NAME
            )
        return cls._instance

    async def add_to_memory(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[Dict] = None,
        source: str = settings.QWEN_SERVICE
    ) -> List[str]:
        """Add text(s) to vector store with metadata."""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            logger.info(f"[{source}] Adding to vector store: {len(texts)} documents")
            if metadata:
                logger.info(f"[{source}] Metadata: {json.dumps(metadata, indent=2)}")
            
            documents = [
                VectorDocument(text=text, metadata=metadata)
                for text in texts
            ]
            
            result = await self.store.add_documents(documents, source)
            logger.info(f"[{source}] Successfully added {len(result)} documents to vector store")
            return result
        except Exception as e:
            logger.error(f"[{source}] Error adding to vector store: {e}")
            raise

    async def search_memory(
        self,
        query: str,
        num_results: int = 5,
        metadata_filter: Optional[Dict] = None,
        source: Optional[str] = None
    ) -> List[Dict]:
        """Search for similar content in vector store."""
        try:
            source_str = source if source else "all"
            logger.info(f"[{source_str}] Searching vector store: {query[:100]}...")
            if metadata_filter:
                logger.info(f"[{source_str}] Metadata filter: {json.dumps(metadata_filter, indent=2)}")
            
            results = await self.store.search_similar(
                query=query,
                num_results=num_results,
                metadata_filter=metadata_filter,
                source_filter=source
            )
            
            logger.info(f"[{source_str}] Found {len(results)} matching documents")
            for i, result in enumerate(results):
                logger.info(f"[{source_str}] Match {i+1} (score: {result.get('score', 'N/A')}): {result['text'][:100]}...")
            
            return results
        except Exception as e:
            logger.error(f"[{source_str}] Error searching vector store: {e}")
            raise

    async def update_memory(
        self,
        document_id: str,
        new_text: Optional[str] = None,
        new_metadata: Optional[Dict] = None,
        source: Optional[str] = None
    ) -> bool:
        """Update existing memory entry."""
        try:
            source_str = source if source else "all"
            logger.info(f"[{source_str}] Updating document {document_id}")
            if new_metadata:
                logger.info(f"[{source_str}] New metadata: {json.dumps(new_metadata, indent=2)}")
            
            result = await self.store.update_document(
                document_id=document_id,
                new_text=new_text,
                new_metadata=new_metadata,
                source=source
            )
            
            logger.info(f"[{source_str}] Document update {'successful' if result else 'failed'}")
            return result
        except Exception as e:
            logger.error(f"[{source_str}] Error updating document: {e}")
            raise

    async def delete_from_memory(
        self,
        document_ids: Union[str, List[str]],
        source: Optional[str] = None
    ) -> int:
        """Delete entries from memory."""
        try:
            if isinstance(document_ids, str):
                document_ids = [document_ids]
            
            source_str = source if source else "all"
            logger.info(f"[{source_str}] Deleting {len(document_ids)} documents")
            
            count = await self.store.delete_documents(document_ids, source)
            logger.info(f"[{source_str}] Successfully deleted {count} documents")
            return count
        except Exception as e:
            logger.error(f"[{source_str}] Error deleting documents: {e}")
            raise

    async def get_from_memory(
        self,
        document_id: str,
        source: Optional[str] = None
    ) -> Optional[Dict]:
        """Retrieve specific memory entry."""
        try:
            source_str = source if source else "all"
            logger.info(f"[{source_str}] Retrieving document {document_id}")
            
            result = await self.store.get_document(document_id, source)
            if result:
                logger.info(f"[{source_str}] Document found: {result['text'][:100]}...")
            else:
                logger.info(f"[{source_str}] Document not found")
            return result
        except Exception as e:
            logger.error(f"[{source_str}] Error retrieving document: {e}")
            raise

    async def clear_memory(self, source: Optional[str] = None):
        """Clear all or source-specific memories."""
        try:
            source_str = source if source else "all"
            logger.info(f"[{source_str}] Clearing vector store")
            await self.store.clear_collection(source)
            logger.info(f"[{source_str}] Vector store cleared successfully")
        except Exception as e:
            logger.error(f"[{source_str}] Error clearing vector store: {e}")
            raise

# Convenience functions for service-specific operations
async def qwen_add_memory(texts: Union[str, List[str]], metadata: Optional[Dict] = None) -> List[str]:
    """Add memory specifically for Qwen service."""
    manager = VectorStoreManager()
    return await manager.add_to_memory(texts, metadata, settings.QWEN_SERVICE)

async def qwen_search_memory(query: str, num_results: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
    """Search memory specifically for Qwen service."""
    manager = VectorStoreManager()
    return await manager.search_memory(query, num_results, metadata_filter, settings.QWEN_SERVICE)

async def autogen_add_memory(texts: Union[str, List[str]], metadata: Optional[Dict] = None) -> List[str]:
    """Add memory specifically for Autogen service."""
    manager = VectorStoreManager()
    return await manager.add_to_memory(texts, metadata, settings.AUTOGEN_SERVICE)

async def autogen_search_memory(query: str, num_results: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
    """Search memory specifically for Autogen service."""
    manager = VectorStoreManager()
    return await manager.search_memory(query, num_results, metadata_filter, settings.AUTOGEN_SERVICE)

# Shared memory operations
async def search_all_memory(query: str, num_results: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
    """Search memory across all services."""
    manager = VectorStoreManager()
    return await manager.search_memory(query, num_results, metadata_filter)

async def clear_all_memory():
    """Clear memory across all services."""
    manager = VectorStoreManager()
    await manager.clear_memory()
