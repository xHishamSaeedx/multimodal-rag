"""
FastAPI application initialization.

This module creates and configures the FastAPI application instance.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import api_router
from app.api.middleware import CorrelationIDMiddleware
from app.utils.logging import get_logger, configure_logging

# Configure structured logging with JSON output
configure_logging(
    log_level="INFO" if not settings.debug else "DEBUG",
    json_output=True,
    include_timestamp=True,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Pre-initializes services at startup:
    - Elasticsearch client
    - Qdrant client
    - Supabase client
    - TextEmbedder (embedding model)
    - ImageEmbedder (embedding model)
    - All retrievers (Sparse, Dense, Table, Image, Hybrid)
    - AnswerGenerator with all dependencies
    
    Everything is pre-warmed and ready to handle queries immediately.
    
    Cleans up resources on shutdown.
    """
    # Startup: Pre-initialize all services
    logger.info(
        "application_startup",
        message="Starting application - Pre-warming all services and models...",
    )
    
    try:
        # Pre-initialize database clients
        logger.info("service_initialization", service="Elasticsearch", status="starting")
        from app.core.database import get_elasticsearch_client
        es_client = get_elasticsearch_client()
        app.state.elasticsearch_client = es_client
        logger.info("service_initialization", service="Elasticsearch", status="ready")
        
        logger.info("service_initialization", service="Qdrant", status="starting")
        from app.core.database import get_qdrant_client
        qdrant_client = get_qdrant_client()
        app.state.qdrant_client = qdrant_client
        logger.info("service_initialization", service="Qdrant", status="ready")
        
        # Pre-initialize Supabase client (needed for AnswerGenerator)
        logger.info("service_initialization", service="Supabase", status="starting")
        from app.core.database import get_supabase_client
        supabase_client = get_supabase_client()
        app.state.supabase_client = supabase_client
        logger.info("service_initialization", service="Supabase", status="ready")
        
        # Pre-initialize embedding models (these are expensive!)
        logger.info("service_initialization", service="TextEmbedder", status="starting")
        from app.services.embedding.text_embedder import TextEmbedder
        text_embedder = TextEmbedder()
        app.state.text_embedder = text_embedder
        logger.info(
            "service_initialization",
            service="TextEmbedder",
            status="ready",
            model_name=text_embedder.model_name,
            embedding_dim=text_embedder.embedding_dim,
        )
        
        logger.info("service_initialization", service="ImageEmbedder", status="starting")
        from app.services.embedding.image_embedder import ImageEmbedder
        image_embedder = ImageEmbedder(model_type="clip")
        app.state.image_embedder = image_embedder
        logger.info(
            "service_initialization",
            service="ImageEmbedder",
            status="ready",
            model_name=image_embedder.model_name,
            embedding_dim=image_embedder.embedding_dim,
            model_type=image_embedder.model_type,
        )
        
        # Pre-initialize captioning processor (for image captioning during ingestion)
        logger.info("service_initialization", service="CaptioningProcessor", status="starting")
        try:
            from app.services.vision import VisionProcessorFactory
            captioning_processor = VisionProcessorFactory.create_processor(mode="captioning")
            # Pre-load the model by calling _load_model() if available
            # This ensures the model is loaded and ready before first use
            if hasattr(captioning_processor, '_load_model'):
                try:
                    captioning_processor._load_model()
                    logger.debug("Pre-loaded captioning model")
                except Exception as load_error:
                    logger.warning(
                        "service_initialization",
                        service="CaptioningProcessor",
                        status="model_load_failed",
                        error=str(load_error),
                        message="Model will be loaded lazily on first use",
                    )
            app.state.captioning_processor = captioning_processor
            model_name = getattr(captioning_processor, 'model_name', 'unknown')
            model_info = captioning_processor.get_model_info() if hasattr(captioning_processor, 'get_model_info') else {}
            logger.info(
                "service_initialization",
                service="CaptioningProcessor",
                status="ready",
                model_name=model_name,
                model_loaded=model_info.get('status') == 'loaded' if model_info else False,
            )
        except Exception as e:
            logger.warning(
                "service_initialization",
                service="CaptioningProcessor",
                status="failed",
                error=str(e),
                message="Captioning will be initialized lazily on first use",
            )
            app.state.captioning_processor = None
        
        # Pre-initialize repositories
        logger.info("service_initialization", service="Repositories", status="starting")
        from app.repositories.vector_repository import VectorRepository
        from app.repositories.sparse_repository import SparseRepository
        vector_repo = VectorRepository(vector_size=text_embedder.embedding_dim)
        sparse_repo = SparseRepository()
        app.state.vector_repository = vector_repo
        app.state.sparse_repository = sparse_repo
        logger.info("service_initialization", service="Repositories", status="ready")
        
        # Pre-initialize all retrievers with pre-warmed models
        logger.info("service_initialization", service="Retrievers", status="starting")
        from app.services.retrieval.dense_retriever import DenseRetriever
        from app.services.retrieval.sparse_retriever import SparseRetriever
        from app.services.retrieval.table_retriever import TableRetriever
        from app.services.retrieval.image_retriever import ImageRetriever
        from app.services.retrieval.hybrid_retriever import HybridRetriever
        
        dense_retriever = DenseRetriever(
            vector_repository=vector_repo,
            embedder=text_embedder
        )
        sparse_retriever = SparseRetriever(sparse_repository=sparse_repo)
        table_retriever = TableRetriever(embedder=text_embedder)
        image_retriever = ImageRetriever(embedder=image_embedder)
        
        # Pre-initialize HybridRetriever with all pre-warmed retrievers
        hybrid_retriever = HybridRetriever(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            table_retriever=table_retriever,
            image_retriever=image_retriever,
        )
        app.state.hybrid_retriever = hybrid_retriever
        logger.info("service_initialization", service="HybridRetriever", status="ready")
        
        # Pre-initialize AnswerGenerator (includes Supabase storage, Groq client, vision processor)
        logger.info("service_initialization", service="AnswerGenerator", status="starting")
        from app.services.generation.answer_generator import AnswerGenerator
        answer_generator = AnswerGenerator()
        app.state.answer_generator = answer_generator
        
        # Log vision processor status
        vision_info = {}
        if answer_generator.vision_processor:
            vision_info = {
                "vision_mode": "vision_llm",
                "vision_provider": getattr(settings, "vision_llm_provider", "unknown"),
                "vision_model": getattr(settings, "vision_llm_model", "unknown"),
                "vision_ready": True,
            }
        else:
            vision_info = {
                "vision_mode": "captioning",
                "vision_ready": False,
            }
        
        logger.info(
            "service_initialization",
            service="AnswerGenerator",
            status="ready",
            model=answer_generator.model,
            **vision_info,
        )
        
        logger.info(
            "application_startup",
            message="All services and models pre-warmed and ready to handle queries!",
            status="ready",
        )
        
    except Exception as e:
        logger.error(
            "application_startup_error",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        # Don't fail startup, but log the error
        # Services will be initialized lazily on first use
        logger.warning(
            "application_startup_fallback",
            message="Some services failed to initialize. They will be initialized lazily on first request",
        )
    
    yield
    
    # Shutdown: Cleanup (if needed)
    logger.info("application_shutdown", message="Shutting down application...")
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
    
    # Add correlation ID middleware (must be before CORS to capture all requests)
    app.add_middleware(CorrelationIDMiddleware)
    
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
