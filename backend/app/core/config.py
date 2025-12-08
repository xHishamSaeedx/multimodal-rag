"""
Application configuration.

This module loads and validates settings from config.yaml and environment variables.
"""
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


# Get the backend directory (parent of app directory)
BACKEND_DIR = Path(__file__).parent.parent.parent
CONFIG_FILE = BACKEND_DIR / "config.yaml"
ENV_FILE = BACKEND_DIR / ".env"


def load_config_yaml() -> dict:
    """Load configuration from config.yaml file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Application settings loaded from config.yaml and environment variables."""

    def __init__(self, **kwargs):
        # Load config from YAML file
        config_data = load_config_yaml()

        # Set defaults from config.yaml where available
        # API Settings
        kwargs.setdefault('api_title', config_data.get('api', {}).get('title', "Multimodal RAG API"))
        kwargs.setdefault('api_version', config_data.get('api', {}).get('version', "1.0.0"))
        kwargs.setdefault('api_description', config_data.get('api', {}).get('description', "API for Multimodal Retrieval-Augmented Generation"))

        # Server Settings
        kwargs.setdefault('host', config_data.get('api', {}).get('host', "0.0.0.0"))
        kwargs.setdefault('port', config_data.get('api', {}).get('port', 8000))

        # CORS Settings
        kwargs.setdefault('cors_origins', config_data.get('api', {}).get('cors_origins', ["http://localhost:3000", "http://localhost:5173"]))

        # Environment
        kwargs.setdefault('environment', config_data.get('app', {}).get('environment', "development"))

        # MinIO Settings
        minio_config = config_data.get('minio', {})
        kwargs.setdefault('minio_endpoint', minio_config.get('endpoint', "localhost:9000"))
        kwargs.setdefault('minio_bucket_name', minio_config.get('bucket_name', "raw-documents"))
        kwargs.setdefault('minio_use_ssl', minio_config.get('use_ssl', False))
        kwargs.setdefault('minio_region', minio_config.get('region'))

        # Qdrant Settings
        qdrant_config = config_data.get('qdrant', {})
        kwargs.setdefault('qdrant_host', qdrant_config.get('host', "localhost"))
        kwargs.setdefault('qdrant_port', qdrant_config.get('port', 6333))
        kwargs.setdefault('qdrant_grpc_port', qdrant_config.get('grpc_port', 6334))
        kwargs.setdefault('qdrant_collection_name', "text_chunks")  # Keep as text_chunks
        kwargs.setdefault('qdrant_vector_size', qdrant_config.get('collections', {}).get('text_chunks', {}).get('vector_size', 768))
        kwargs.setdefault('qdrant_timeout', qdrant_config.get('timeout', 30))

        # Elasticsearch Settings
        es_config = config_data.get('elasticsearch', {})
        kwargs.setdefault('elasticsearch_url', es_config.get('url', "http://localhost:9200"))
        kwargs.setdefault('elasticsearch_index_name', es_config.get('index_name', "chunks"))
        kwargs.setdefault('elasticsearch_timeout', es_config.get('timeout', 30))

        # Embedding Settings
        embedding_config = config_data.get('embeddings', {})
        kwargs.setdefault('embedding_model', embedding_config.get('model', "intfloat/e5-base-v2"))
        kwargs.setdefault('embedding_device', embedding_config.get('device', "cpu"))
        kwargs.setdefault('embedding_batch_size', embedding_config.get('batch_size', 32))

        # Vision Processing Settings
        vision_config = config_data.get('vision', {})
        kwargs.setdefault('vision_processing_mode', vision_config.get('processing_mode', "captioning"))
        kwargs.setdefault('vision_llm_provider', vision_config.get('llm_provider', "openai"))
        kwargs.setdefault('vision_llm_model', vision_config.get('llm_model', "gpt-4o"))
        kwargs.setdefault('captioning_model', vision_config.get('captioning_model', "Salesforce/blip-image-captioning-base"))

        # Image Embedding Settings
        image_embedding_config = vision_config.get('image_embedding', {})
        kwargs.setdefault('image_embedding_model_type', image_embedding_config.get('model_type', "clip"))
        kwargs.setdefault('image_embedding_model_name', image_embedding_config.get('model_name', "sentence-transformers/clip-ViT-L-14"))

        # Neo4j Settings
        neo4j_config = config_data.get('neo4j', {})
        kwargs.setdefault('neo4j_enabled', neo4j_config.get('enabled', True))
        kwargs.setdefault('neo4j_uri', neo4j_config.get('uri', "bolt://localhost:7687"))
        kwargs.setdefault('neo4j_user', neo4j_config.get('user', "neo4j"))
        kwargs.setdefault('neo4j_database', neo4j_config.get('database', "neo4j"))
        kwargs.setdefault('neo4j_timeout', neo4j_config.get('timeout', 30))
        kwargs.setdefault('neo4j_max_connection_pool_size', neo4j_config.get('max_connection_pool_size', 50))

        # Initialize with merged kwargs
        super().__init__(**kwargs)

    # API Settings
    api_title: str
    api_version: str
    api_description: str

    # Server Settings
    host: str
    port: int
    debug: bool = False

    # CORS Settings
    cors_origins: list[str]

    # Environment
    environment: str

    # MinIO (S3-compatible) Settings - access/secret keys still from env
    minio_endpoint: str
    minio_access_key: str = "admin"
    minio_secret_key: str = "admin12345"
    minio_bucket_name: str
    minio_use_ssl: bool
    minio_region: Optional[str]

    # Supabase (PostgreSQL) Settings - all from env
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None

    # Qdrant (Vector DB) Settings
    qdrant_host: str
    qdrant_port: int
    qdrant_grpc_port: int
    qdrant_collection_name: str
    qdrant_vector_size: int
    qdrant_timeout: int

    # Elasticsearch (BM25 Sparse Index) Settings
    elasticsearch_url: str
    elasticsearch_index_name: str
    elasticsearch_timeout: int

    # Embedding Settings
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int

    # Groq Settings (for Answer Generation) - API key and model from env
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None

    # Vision Processing Settings
    vision_processing_mode: str
    vision_llm_provider: str
    vision_llm_model: str
    captioning_model: str

    # Image Embedding Settings
    image_embedding_model_type: str
    image_embedding_model_name: str

    # Vision LLM API Keys - from env
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Neo4j (Knowledge Graph) Settings - password from env
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str = "neo4j-password"
    neo4j_database: str
    neo4j_timeout: int
    neo4j_max_connection_pool_size: int
    neo4j_enabled: bool

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
