"""
Configuration management for the Policy Document Summarization Assistant.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    
    # AWS S3 Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_bucket: str = "policy-documents"
    aws_endpoint_url: Optional[str] = None  # For local S3-compatible services
    
    # Vector Database Configuration
    vector_db: str = "faiss"  # Options: "faiss" or "mongo"
    mongodb_uri: Optional[str] = None
    mongodb_db_name: str = "policy_docs"
    mongodb_collection: str = "embeddings"
    faiss_index_path: str = "./data/faiss_index"
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 500
    max_chunk_size: int = 1500
    
    # Summarization Configuration
    summary_temperature: float = 0.3
    max_tokens: int = 2000
    
    # QA/Hallucination Check Configuration
    similarity_threshold: float = 0.7
    
    # Application Configuration
    app_name: str = "Policy Document Summarization Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    max_file_size_mb: int = 30
    
    # Storage paths
    local_storage_path: str = "./data/documents"
    chunks_storage_path: str = "./data/chunks"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

