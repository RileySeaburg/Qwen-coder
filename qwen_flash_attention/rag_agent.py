import logging
from typing import List, Dict, Any, Optional, Union, AsyncIterator, cast
from typing_extensions import TypedDict
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from .schemas import Message, Agent
import json
import os
import time
import torch
import numpy as np
from numpy.typing import NDArray
from fastapi import Request

logger = logging.getLogger(__name__)

class SearchResult(TypedDict):
    text: str
    metadata: Dict[str, Any]
    score: float

class MongoDocument(TypedDict):
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

class MongoIndex(TypedDict):
    name: str
    key: List[tuple]

class RAGAgent:
    """RAG-enhanced agent using MongoDB for storage and retrieval."""
    
    def __init__(
        self,
        mongodb_url: str = "mongodb://localhost:27017",
        database_name: str = "shared_vectors",
        collection_name: str = "embeddings",
        wait_until_ready: float = 120.0,
        request: Optional[Request] = None,
        qwen_model: Optional[Any] = None
    ):
        try:
            logger.info(f"Connecting to MongoDB at {mongodb_url}")
            self.client = AsyncIOMotorClient(
                mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            self.db = self.client[database_name]
            self.collection: AsyncIOMotorCollection = self.db[collection_name]
            self.wait_until_ready = wait_until_ready
            self.request = request
            self.qwen_model = qwen_model
            self.initialized = False
            logger.info("MongoDB client initialized")
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {e}", exc_info=True)
            self.client = None
            self.db = None
            self.collection = None

    async def initialize(self):
        """Initialize MongoDB connection and indexes."""
        if not self.initialized:
            try:
                logger.info("Initializing RAG agent...")

                if self.client is None:
                    raise RuntimeError("MongoDB client not initialized")
                
                # Ping the server to verify connection
                try:
                    await self.client.admin.command('ping')
                    logger.info("MongoDB connection successful")
                except Exception as e:
                    logger.error(f"MongoDB ping failed: {e}", exc_info=True)
                    raise
                
                # Create text search index if it doesn't exist
                try:
                    indexes: List[MongoIndex] = await self.collection.list_indexes().to_list(None)
                    has_text_index = any(idx.get('name') == 'text_search' for idx in indexes)
                    
                    if not has_text_index:
                        await self.collection.create_index(
                            [("text", "text"), ("metadata.source", 1)],
                            name="text_search"
                        )
                        logger.info("Created text search index")
                except Exception as e:
                    logger.error(f"Error creating text index: {e}", exc_info=True)
                    raise

                # Create embedding index if it doesn't exist
                try:
                    has_embedding_index = any(idx.get('name') == 'embedding_index' for idx in indexes)
                    if not has_embedding_index:
                        await self.collection.create_index([("embedding", 1)], name="embedding_index")
                        logger.info("Created embedding index")
                except Exception as e:
                    logger.error(f"Error creating embedding index: {e}", exc_info=True)
                    raise

                # Create metadata indexes
                try:
                    await self.collection.create_index([("metadata.source", 1)])
                    await self.collection.create_index([("metadata.timestamp", 1)])
                    logger.info("Created metadata indexes")
                except Exception as e:
                    logger.error(f"Error creating metadata indexes: {e}", exc_info=True)
                    raise

                self.initialized = True
                logger.info("RAG agent initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing RAG agent: {e}", exc_info=True)
                self.initialized = False
                raise

    def set_qwen_model(self, model: Any):
        """Set the Qwen model instance."""
        self.qwen_model = model
        logger.info("Qwen model set in RAG agent")

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using the server's model."""
        if self.qwen_model is None:
            if self.request and hasattr(self.request.app.state, "qwen_model"):
                self.qwen_model = self.request.app.state.qwen_model
            else:
                raise RuntimeError("Qwen model not available")
        
        return self.qwen_model.get_embeddings(texts)

    async def search_similar(
        self,
        query: str,
        num_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar documents using cosine similarity."""
        try:
            if not self.initialized:
                await self.initialize()

            if self.collection is None:
                logger.warning("MongoDB not available, returning empty results")
                return []

            logger.info(f"Searching for: {query[:100]}...")

            # Generate query embedding
            try:
                query_embedding = (await self.get_embeddings([query]))[0]
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                return []

            # Build match conditions
            match_conditions: Dict[str, Any] = {}
            if metadata_filter:
                for key, value in metadata_filter.items():
                    match_conditions[f"metadata.{key}"] = value
            if source_filter:
                match_conditions["metadata.source"] = source_filter

            # Find documents
            cursor = self.collection.find(match_conditions)
            documents = await cursor.to_list(length=None)

            # Calculate cosine similarity
            results: List[SearchResult] = []
            for doc in documents:
                doc_embedding = np.array(doc["embedding"])
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(similarity)
                })

            # Sort by similarity and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:num_results]

            logger.info(f"Found {len(results)} matching documents")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            return []

    async def enhance_prompt(
        self,
        messages: List[Message],
        agent: Agent
    ) -> List[Message]:
        """Enhance agent prompt with relevant knowledge."""
        try:
            if not self.initialized:
                await self.initialize()

            # Get the latest user message
            user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
            if not user_message:
                return messages

            # Search for relevant knowledge
            try:
                knowledge = await self.search_similar(
                    query=user_message.content,
                    metadata_filter={"agent_role": agent.role}
                )
            except Exception as e:
                logger.error(f"Error searching knowledge: {e}")
                return messages

            if not knowledge:
                return messages

            # Format knowledge as context
            context = "Relevant information from knowledge base:\n\n"
            for i, item in enumerate(knowledge, 1):
                context += f"{i}. {item['text']}\n"
                if item['metadata'].get('source'):
                    context += f"   Source: {item['metadata']['source']}\n"
                context += "\n"

            # Insert context before user message
            enhanced_messages = messages.copy()
            insert_idx = next(
                (i for i, msg in enumerate(enhanced_messages) if msg == user_message),
                len(enhanced_messages)
            )
            enhanced_messages.insert(insert_idx, Message(
                role="system",
                content=context,
                name="knowledge_base"
            ))

            logger.info(f"Enhanced prompt with {len(knowledge)} knowledge items")
            return enhanced_messages
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return messages

    def __del__(self):
        """Close MongoDB connection on deletion."""
        try:
            if self.client is not None:
                self.client.close()
                logger.info("MongoDB connection closed")
        except:
            pass
