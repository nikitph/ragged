"""
Main API router
"""

from fastapi import APIRouter

from ragged.api.v1.endpoints import health, videos, search

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(search.router, prefix="/search", tags=["search"])

# TODO: Add tenant and admin routers
# api_router.include_router(tenants.router, prefix="/tenants", tags=["tenants"])
# api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
