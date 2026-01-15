import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    
    llm_model_name: str = Field(
        default="qwen/qwen3-32b",
        validation_alias="LLM_MODEL_NAME"
    )
    llm_temperature: float = Field(default=0.0, validation_alias="LLM_TEMPERATURE")
    llm_rate_limit_rps: float = Field(
        default=0.1,
        validation_alias="LLM_RATE_LIMIT_RPS",
        description="Requests per second for LLM rate limiting"
    )
    llm_rate_limit_bucket_size: int = Field(
        default=10,
        validation_alias="LLM_RATE_LIMIT_BUCKET_SIZE"
    )
    
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL_NAME"
    )
    embedding_device: str = Field(default="cpu", validation_alias="EMBEDDING_DEVICE")
    
    chunk_size: int = Field(default=4000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    
    rag_top_k: int = Field(
        default=5,
        validation_alias="RAG_TOP_K",
        description="Number of documents to retrieve for RAG"
    )
    
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_title: str = Field(
        default="Agentic RAG API",
        validation_alias="API_TITLE"
    )
    api_version: str = Field(default="0.1.0", validation_alias="API_VERSION")
    
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
