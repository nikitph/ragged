"""
Fixed router to resolve endpoint conflicts
"""

from fastapi import APIRouter

from ragged.api.v1.endpoints import health, videos, search
from ragged.api.v1.endpoints import pdf_upload

api_router = APIRouter()

# Include endpoint routers with specific prefixes to avoid conflicts
api_router.include_router(health.router, prefix="/health", tags=["health"])

# Original videos endpoints (for general video operations)
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])

# Search endpoints
api_router.include_router(search.router, prefix="/search", tags=["search"])

# PDF processing endpoints with specific prefix to avoid conflicts
api_router.include_router(pdf_upload.router, prefix="/pdf", tags=["pdf-processing"])

# TODO: Add tenant and admin routers
# api_router.include_router(tenants.router, prefix="/tenants", tags=["tenants"])
# api_router.include_router(admin.router, prefix="/admin", tags=["admin"])