"""
Application configuration.

This module loads and validates environment variables and application settings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


# Get the backend directory (parent of app directory)
BACKEND_DIR = Path(__file__).parent.parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    api_title: str = "Multimodal RAG API"
    api_version: str = "1.0.0"
    api_description: str = "API for Multimodal Retrieval-Augmented Generation"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS Settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Environment
    environment: str = "development"
    
    # MinIO (S3-compatible) Settings
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "admin"
    minio_secret_key: str = "admin12345"
    minio_bucket_name: str = "raw-documents"
    minio_use_ssl: bool = False
    minio_region: Optional[str] = None
    
    # Supabase (PostgreSQL) Settings
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    
    # Qdrant (Vector DB) Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection_name: str = "text_chunks"
    qdrant_vector_size: int = 768  # e5-base-v2 uses 768 dimensions
    qdrant_timeout: int = 30
    
    # Elasticsearch (BM25 Sparse Index) Settings
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index_name: str = "chunks"
    elasticsearch_timeout: int = 30
    
    # Embedding Settings
    embedding_model: str = "intfloat/e5-base-v2"  # Recommended model (768 dim)
    embedding_device: str = "cpu"  # "cpu" or "cuda"
    embedding_batch_size: int = 32
    
    # Groq Settings (for Answer Generation)
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None  # Will be loaded from env file
    
    # Vision Processing Settings
    vision_processing_mode: str = "captioning"  # "captioning" or "vision_llm"
    vision_llm_provider: str = "openai"  # "openai" or "google"
    vision_llm_model: str = "gpt-4o"  # Model name (OpenAI: gpt-4o or gpt-4o-mini, Google: gemini-1.5-pro or gemini-1.5-flash)
    captioning_model: str = "Salesforce/blip-image-captioning-base"  # BLIP model for captioning
    
    # Vision LLM API Keys
    openai_api_key: Optional[str] = None  # Required when vision_llm_provider="openai"
    google_api_key: Optional[str] = None  # Required when vision_llm_provider="google"
    
    # Note: These can be overridden via environment variables:
    # VISION_PROCESSING_MODE, VISION_LLM_PROVIDER, VISION_LLM_MODEL, CAPTIONING_MODEL
    # OPENAI_API_KEY, GOOGLE_API_KEY
    
    # Observability Settings (Optional)
    loki_enabled: bool = False  # Whether to push logs directly to Loki (optional, Promtail handles collection)
    loki_url: str = "http://localhost:3100"  # Loki API URL
    grafana_url: str = "http://localhost:3001"  # Grafana UI URL
    
    model_config = {
        "env_file": str(ENV_FILE),  # Use .env file in backend directory
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields in .env file that aren't defined here
    }


settings = Settings()
