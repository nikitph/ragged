"""
Search endpoints
TODO: Add tenant-scoped search
"""
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[str]
    query_time_ms: float

@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search across video knowledge base"""
    # TODO: Implement actual search
    return SearchResponse(
        results=[f"Mock result {i} for query: {request.query}" for i in range(request.top_k)],
        query_time_ms=42.0
    )
