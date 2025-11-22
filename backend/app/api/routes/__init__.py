"""API route handlers."""
from fastapi import APIRouter
from app.api.routes import health, ingest, documents, query

api_router = APIRouter(prefix="/api/v1")

# Include all route modules
api_router.include_router(health.router)
api_router.include_router(ingest.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
