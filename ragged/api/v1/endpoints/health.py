"""
Health check endpoints
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "healthy"}

@router.get("/detailed")
async def detailed_health():
    # TODO: Add database, redis, storage health checks
    return {
        "status": "healthy",
        "database": "connected",
        "redis": "connected",
        "storage": "available"
    }
