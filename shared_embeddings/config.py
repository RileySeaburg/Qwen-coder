from pydantic_settings import BaseSettings
from typing import Optional

class VectorStoreSettings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "shared_vectors"
    COLLECTION_NAME: str = "embeddings"
    MODEL_NAME: str = "Qwen/Qwen2.5-Coder-3B"
    
    # Service identifiers
    QWEN_SERVICE: str = "qwen"
    AUTOGEN_SERVICE: str = "autogen"
    
    class Config:
        env_prefix = "VECTOR_"
        case_sensitive = True

settings = VectorStoreSettings()
