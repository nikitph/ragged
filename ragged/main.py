"""
Ragged - Video-based RAG as a Service
Main FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragged.api.v1.router import api_router
from ragged.core.config import settings

app = FastAPI(
    title="Ragged",
    description="Video-based RAG as a Service",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Ragged - Video RAG Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ragged"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
