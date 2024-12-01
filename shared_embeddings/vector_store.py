from typing import List, Dict, Optional, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from pydantic import BaseModel
import torch
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDocument(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

class VectorStore:
    def __init__(
        self,
        mongodb_url: str = "mongodb://localhost:27017",
        database_name: str = "shared_vectors",
        collection_name: str = "embeddings",
        model_name: str = "Qwen/Qwen2.5-Coder-3B"
    ):
        logger.info(f"Initializing VectorStore with URL: {mongodb_url}, DB: {database_name}, Collection: {collection_name}")
        self.client = AsyncIOMotorClient(mongodb_url)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False

    async def initialize(self):
        """Initialize database and indexes."""
        if not self.initialized:
            logger.info("Starting vector store initialization...")
            await self._initialize_db()
            self.initialized = True
            logger.info("Vector store initialization complete")

    async def _initialize_db(self):
        """Initialize database and collection if they don't exist."""
        try:
            # List all databases to check if ours exists
            databases = await self.client.list_database_names()
            logger.info(f"Existing databases: {databases}")
            
            if self.db.name not in databases:
                logger.info(f"Creating database: {self.db.name}")
                # Create a dummy document to initialize the database
                await self.collection.insert_one({"_id": "init", "type": "initialization"})
                await self.collection.delete_one({"_id": "init"})
                logger.info("Database created successfully")

            # Create necessary indexes
            await self._ensure_indexes()
            
            # Verify collection exists
            collections = await self.db.list_collection_names()
            logger.info(f"Collections in {self.db.name}: {collections}")
            
            # Log collection stats
            stats = await self.db.command("collstats", self.collection.name)
            logger.info(f"Collection stats: {json.dumps(stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    async def _ensure_indexes(self):
        """Create necessary indexes for vector search."""
        try:
            logger.info("Creating indexes...")
            
            # Get existing indexes
            existing_indexes = await self.collection.list_indexes().to_list(None)
            existing_index_names = [idx['name'] for idx in existing_indexes]
            logger.info(f"Existing indexes: {existing_index_names}")

            # Create text search index if it doesn't exist
            if "text_search" not in existing_index_names:
                logger.info("Creating text search index...")
                await self.collection.create_index(
                    [("text", "text"), ("metadata.source", 1)],
                    name="text_search"
                )
                logger.info("Text search index created successfully")

            # Create metadata source index if it doesn't exist
            if "metadata.source_1" not in existing_index_names:
                logger.info("Creating metadata source index...")
                await self.collection.create_index(
                    [("metadata.source", 1)],
                    name="metadata.source_1",
                    background=True
                )
                logger.info("Metadata source index created successfully")

            # Create metadata timestamp index if it doesn't exist
            if "metadata.timestamp_1" not in existing_index_names:
                logger.info("Creating metadata timestamp index...")
                await self.collection.create_index(
                    [("metadata.timestamp", 1)],
                    name="metadata.timestamp_1",
                    background=True
                )
                logger.info("Metadata timestamp index created successfully")

            # Note: Vector search index must be created through Atlas UI or API
            logger.info("Note: Vector search index must be created through Atlas UI")

            # Verify indexes after creation
            final_indexes = await self.collection.list_indexes().to_list(None)
            logger.info(f"Final indexes: {[idx['name'] for idx in final_indexes]}")

        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    async def add_documents(self, documents: List[VectorDocument], source: str) -> List[str]:
        """Add documents to the vector store with source tracking."""
        try:
            if not self.initialized:
                await self.initialize()

            logger.info(f"Adding {len(documents)} documents from source: {source}")
            texts = [doc.text for doc in documents]
            embeddings = self.embedding_model.encode(texts)

            docs_to_insert = []
            for doc, embedding in zip(documents, embeddings):
                doc_dict = doc.dict()
                doc_dict["embedding"] = embedding.tolist()
                doc_dict["metadata"] = doc_dict.get("metadata", {})
                doc_dict["metadata"].update({
                    "source": source,
                    "timestamp": time.time()
                })
                docs_to_insert.append(doc_dict)

            result = await self.collection.insert_many(docs_to_insert)
            logger.info(f"Successfully added {len(result.inserted_ids)} documents")
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def search_similar(
        self,
        query: str,
        num_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with optional source filtering."""
        try:
            if not self.initialized:
                await self.initialize()

            logger.info(f"Searching for: {query[:100]}...")
            query_embedding = self.embedding_model.encode(query)

            # Build aggregation pipeline
            pipeline: List[Dict[str, Any]] = [
                {
                    "$search": {
                        "index": "default",  # Use Atlas Search index
                        "compound": {
                            "should": [
                                {
                                    "text": {
                                        "query": query,
                                        "path": "text",
                                        "score": {"boost": {"value": 1.5}}
                                    }
                                },
                                {
                                    "knnBeta": {
                                        "vector": query_embedding.tolist(),
                                        "path": "embedding",
                                        "k": num_results * 2
                                    }
                                }
                            ]
                        }
                    }
                }
            ]

            # Add filters if provided
            match_conditions: Dict[str, Any] = {}
            if metadata_filter:
                for key, value in metadata_filter.items():
                    match_conditions[f"metadata.{key}"] = value
            if source_filter:
                match_conditions["metadata.source"] = source_filter

            if match_conditions:
                pipeline.append({"$match": match_conditions})

            # Limit results
            pipeline.append({"$limit": num_results})

            # Project relevant fields
            pipeline.append({
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "searchScore"}
                }
            })

            logger.info(f"Search pipeline: {json.dumps(pipeline, indent=2)}")
            results = []
            async for doc in self.collection.aggregate(pipeline):
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": doc.get("score", 0.0)  # Provide default score if missing
                })

            logger.info(f"Found {len(results)} matching documents")
            return results
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise

    async def update_document(
        self,
        document_id: str,
        new_text: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> bool:
        """Update a document with source verification."""
        try:
            if not self.initialized:
                await self.initialize()

            # Build query with optional source check
            query: Dict[str, Any] = {"_id": document_id}
            if source is not None:  # Only add source to query if it's provided
                query["metadata.source"] = source

            update_dict: Dict[str, Any] = {}
            if new_text is not None:  # Only update text if provided
                new_embedding = self.embedding_model.encode(new_text)
                update_dict.update({
                    "text": new_text,
                    "embedding": new_embedding.tolist()
                })
            
            if new_metadata is not None:  # Only update metadata if provided
                update_dict["metadata"] = new_metadata
                update_dict["metadata"]["timestamp"] = time.time()

            if update_dict:
                result = await self.collection.update_one(
                    query,
                    {"$set": update_dict}
                )
                return result.modified_count > 0
            return False
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise

    async def delete_documents(
        self,
        document_ids: List[str],
        source: Optional[str] = None
    ) -> int:
        """Delete documents with optional source verification."""
        try:
            if not self.initialized:
                await self.initialize()

            query: Dict[str, Any] = {"_id": {"$in": document_ids}}
            if source is not None:  # Only add source to query if it's provided
                query["metadata.source"] = source

            result = await self.collection.delete_many(query)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    async def get_document(
        self,
        document_id: str,
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document with optional source verification."""
        try:
            if not self.initialized:
                await self.initialize()

            query: Dict[str, Any] = {"_id": document_id}
            if source is not None:  # Only add source to query if it's provided
                query["metadata.source"] = source

            doc = await self.collection.find_one(query)
            if doc:
                return {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "embedding": doc.get("embedding")
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            raise

    async def clear_collection(self, source: Optional[str] = None):
        """Clear documents with optional source filtering."""
        try:
            if not self.initialized:
                await self.initialize()

            query: Dict[str, Any] = {}
            if source is not None:  # Only add source to query if it's provided
                query["metadata.source"] = source
            await self.collection.delete_many(query)
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def __del__(self):
        """Close MongoDB connection on deletion."""
        try:
            self.client.close()
            logger.info("MongoDB connection closed")
        except:
            pass
