"""
Backend entry point.

This is the main entry point for running the FastAPI application.
Run with: uvicorn main:app --reload
"""
from app.api.main import app

__all__ = ["app"]
