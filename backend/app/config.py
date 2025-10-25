# config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    OPENAI_MODEL: str = "gpt-5-nano-2025-08-07"  # Updated to GPT-4 Omni
    OPENAI_VISION_MODEL: str = "gpt-5-nano-2025-08-07"  # GPT-4 Omni has vision built-in
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: Optional[str] = None
    
    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "docbot"
    POSTGRES_USER: str = "docbot_user"
    POSTGRES_PASSWORD: str = "secure_password"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # MinIO Configuration
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_SECURE: bool = False
    
    # Application Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_RESULTS: int = 10
    EMBEDDING_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"

settings = Settings()