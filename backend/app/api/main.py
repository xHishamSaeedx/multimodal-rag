"""
FastAPI application initialization.

This module creates and configures the FastAPI application instance.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Pre-initializes services at startup:
    - Elasticsearch client
    - Qdrant client
    - TextEmbedder (embedding model)
    - HybridRetriever and its dependencies
    
    Cleans up resources on shutdown.
    """
    # Startup: Pre-initialize all services
    logger.info("=" * 60)
    logger.info("Starting application - Pre-warming services...")
    logger.info("=" * 60)
    
    try:
        # Pre-initialize database clients
        logger.info("Pre-initializing Elasticsearch client...")
        from app.core.database import get_elasticsearch_client
        es_client = get_elasticsearch_client()
        app.state.elasticsearch_client = es_client
        logger.info("✓ Elasticsearch client ready")
        
        logger.info("Pre-initializing Qdrant client...")
        from app.core.database import get_qdrant_client
        qdrant_client = get_qdrant_client()
        app.state.qdrant_client = qdrant_client
        logger.info("✓ Qdrant client ready")
        
        # Pre-initialize embedding model (this is the expensive one!)
        logger.info("Pre-loading embedding model (this may take a moment)...")
        from app.services.embedding.text_embedder import TextEmbedder
        text_embedder = TextEmbedder()
        app.state.text_embedder = text_embedder
        logger.info(f"✓ Embedding model loaded: {text_embedder.model_name} (dim: {text_embedder.embedding_dim})")
        
        # Pre-initialize repositories
        logger.info("Pre-initializing repositories...")
        from app.repositories.vector_repository import VectorRepository
        from app.repositories.sparse_repository import SparseRepository
        vector_repo = VectorRepository(vector_size=text_embedder.embedding_dim)
        sparse_repo = SparseRepository()
        app.state.vector_repository = vector_repo
        app.state.sparse_repository = sparse_repo
        logger.info("✓ Repositories ready")
        
        # Pre-initialize retrievers
        logger.info("Pre-initializing retrievers...")
        from app.services.retrieval.dense_retriever import DenseRetriever
        from app.services.retrieval.sparse_retriever import SparseRetriever
        from app.services.retrieval.hybrid_retriever import HybridRetriever
        
        dense_retriever = DenseRetriever(
            vector_repository=vector_repo,
            embedder=text_embedder
        )
        sparse_retriever = SparseRetriever(sparse_repository=sparse_repo)
        hybrid_retriever = HybridRetriever(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever
        )
        app.state.hybrid_retriever = hybrid_retriever
        logger.info("✓ HybridRetriever ready")
        
        logger.info("=" * 60)
        logger.info("All services pre-warmed and ready!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to pre-warm services: {str(e)}", exc_info=True)
        # Don't fail startup, but log the error
        # Services will be initialized lazily on first use
        logger.warning("Services will be initialized lazily on first request")
    
    yield
    
    # Shutdown: Cleanup (if needed)
    logger.info("Shutting down application...")
    # Clients and models will be cleaned up automatically


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Multimodal RAG API",
            "version": settings.api_version,
            "docs": "/docs"
        }
    
    return app


# Create the app instance
app = create_app()
