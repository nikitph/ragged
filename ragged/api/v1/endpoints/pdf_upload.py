"""
PDF upload endpoint with path-safe video encoding
"""

import os
import asyncio
import aiofiles
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
import tempfile
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
import io

# Import our video encoder components
try:
    from ragged.video.encoder import TextVectorPipeline, VectorMP4Encoder, create_text_vector_mp4
except ImportError as e:
    print(f"Warning: Video encoder not available: {e}")
    # Define dummy classes for development
    class TextVectorPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def process_documents(self, docs):
            return [], []

    class VectorMP4Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def add_vectors(self, *args, **kwargs):
            pass
        def encode_to_mp4(self, path):
            Path(path).touch()  # Create empty file

router = APIRouter()

# Global storage for tracking processing status
processing_status: Dict[str, Dict[str, Any]] = {}

class ProcessingStage(str, Enum):
    UPLOADING = "uploading"
    EXTRACTING_TEXT = "extracting_text"
    GENERATING_VECTORS = "generating_vectors"
    BUILDING_FAISS = "building_faiss"
    CREATING_MANIFEST = "creating_manifest"
    GENERATING_MP4 = "generating_mp4"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingStatus(BaseModel):
    job_id: str
    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str
    error: Optional[str] = None
    files: Optional[Dict[str, str]] = None  # Generated file paths
    started_at: datetime
    completed_at: Optional[datetime] = None

class UploadResponse(BaseModel):
    job_id: str
    message: str
    status_url: str

def get_storage_dir() -> Path:
    """Get the storage directory for uploaded files"""
    storage_dir = Path("./storage/uploads").resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text content from PDF bytes"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            except Exception as e:
                print(f"Warning: Could not extract text from page: {e}")
                continue

        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

async def process_pdf_to_video_safe(job_id: str, pdf_content: bytes, filename: str):
    """
    Background task to process PDF and generate video files with path safety
    """
    temp_dir = None
    try:
        # Update status: extracting text
        processing_status[job_id].update({
            "stage": ProcessingStage.EXTRACTING_TEXT,
            "progress": 0.1,
            "message": "Extracting text from PDF..."
        })

        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_content)

        if not text_content.strip():
            raise ValueError("No text content could be extracted from the PDF")

        print(f"Extracted {len(text_content)} characters from PDF")

        # Create document structure for processing
        documents = [{
            "text": text_content,
            "source": filename
        }]

        # Update status: generating vectors
        processing_status[job_id].update({
            "stage": ProcessingStage.GENERATING_VECTORS,
            "progress": 0.3,
            "message": "Generating vector embeddings..."
        })

        # Initialize text-vector pipeline
        pipeline = TextVectorPipeline(
            model_name="all-MiniLM-L6-v2",
            chunk_size=512,
            chunk_overlap=50,
            vector_dim=384
        )

        # Process documents into vectors
        vectors, metadata = pipeline.process_documents(documents)

        if len(vectors) == 0:
            raise ValueError("No vectors generated from document content")

        print(f"Generated {len(vectors)} vectors")

        # Update status: building FAISS index
        processing_status[job_id].update({
            "stage": ProcessingStage.BUILDING_FAISS,
            "progress": 0.5,
            "message": f"Building FAISS index for {len(vectors)} vectors..."
        })

        # Initialize MP4 encoder
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=1000)
        encoder.add_vectors(vectors, metadata)

        # Create temporary directory for safe processing
        temp_dir = tempfile.mkdtemp(prefix="ragged_video_")
        temp_path = Path(temp_dir)

        # Process in temp directory first to avoid path issues
        base_filename = f"{job_id}_{Path(filename).stem}"
        temp_video_file = temp_path / f"{base_filename}.mp4"

        # Update status: creating manifest
        processing_status[job_id].update({
            "stage": ProcessingStage.CREATING_MANIFEST,
            "progress": 0.7,
            "message": "Creating manifest and index files..."
        })

        # Update status: generating MP4
        processing_status[job_id].update({
            "stage": ProcessingStage.GENERATING_MP4,
            "progress": 0.8,
            "message": "Generating MP4 video file..."
        })

        # Encode to MP4 in temp directory first
        encoder.encode_to_mp4(str(temp_video_file))

        # Now move files to final storage location
        storage_dir = get_storage_dir()
        final_video_file = storage_dir / f"{base_filename}.mp4"
        final_manifest_file = storage_dir / f"{base_filename}_manifest.json"
        final_faiss_file = storage_dir / f"{base_filename}_faiss.index"

        # Move generated files to storage
        generated_files = {}

        if temp_video_file.exists():
            shutil.move(str(temp_video_file), str(final_video_file))
            generated_files["mp4"] = str(final_video_file.relative_to(Path.cwd()))
            print(f"Moved video file: {final_video_file}")

        temp_manifest = temp_path / f"{base_filename}_manifest.json"
        if temp_manifest.exists():
            shutil.move(str(temp_manifest), str(final_manifest_file))
            generated_files["manifest"] = str(final_manifest_file.relative_to(Path.cwd()))
            print(f"Moved manifest file: {final_manifest_file}")

        temp_faiss = temp_path / f"{base_filename}_faiss.index"
        if temp_faiss.exists():
            shutil.move(str(temp_faiss), str(final_faiss_file))
            generated_files["faiss"] = str(final_faiss_file.relative_to(Path.cwd()))
            print(f"Moved FAISS file: {final_faiss_file}")

        # Update status: completed
        processing_status[job_id].update({
            "stage": ProcessingStage.COMPLETED,
            "progress": 1.0,
            "message": f"Processing completed successfully. Generated {len(generated_files)} files.",
            "files": generated_files,
            "completed_at": datetime.now()
        })

        print(f"Successfully processed PDF {filename} -> {len(generated_files)} files")

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing PDF {filename}: {error_msg}")

        # Update status: failed
        processing_status[job_id].update({
            "stage": ProcessingStage.FAILED,
            "progress": 0.0,
            "message": "Processing failed",
            "error": error_msg,
            "completed_at": datetime.now()
        })

    finally:
        # Clean up temp directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp directory: {cleanup_error}")

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF file and process it into video format with FAISS index

    Returns:
        job_id and status URL for tracking progress
    """

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Validate file size (50MB limit)
    max_size = 50 * 1024 * 1024  # 50MB

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    try:
        # Read file content
        pdf_content = await file.read()

        if len(pdf_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            )

        if len(pdf_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )

        # Initialize processing status
        processing_status[job_id] = {
            "job_id": job_id,
            "stage": ProcessingStage.UPLOADING,
            "progress": 0.0,
            "message": f"Upload completed. File size: {len(pdf_content):,} bytes",
            "error": None,
            "files": None,
            "started_at": datetime.now(),
            "completed_at": None
        }

        # Start background processing with safe path handling
        background_tasks.add_task(
            process_pdf_to_video_safe,
            job_id,
            pdf_content,
            file.filename
        )

        return UploadResponse(
            job_id=job_id,
            message=f"PDF upload successful. Processing started for {file.filename}",
            status_url=f"/api/v1/videos/status/{job_id}"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process upload: {str(e)}"
        )

@router.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """
    Get the current processing status for a job
    """
    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    status = processing_status[job_id]
    return ProcessingStatus(**status)

@router.get("/list-jobs")
async def list_processing_jobs():
    """
    List all processing jobs (for debugging/admin)
    """
    jobs_list = []
    for job_id, status in processing_status.items():
        jobs_list.append({
            "job_id": job_id,
            "stage": status["stage"],
            "progress": status["progress"],
            "started_at": status["started_at"]
        })

    return {
        "jobs": [job["job_id"] for job in jobs_list],
        "total": len(jobs_list),
        "details": jobs_list
    }

@router.delete("/status/{job_id}")
async def cleanup_job_status(job_id: str):
    """
    Clean up job status (for cleanup/admin)
    """
    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    del processing_status[job_id]
    return {"message": f"Job {job_id} status cleaned up"}

@router.get("/download/{job_id}/{file_type}")
async def download_generated_file(job_id: str, file_type: str):
    """
    Download generated files (mp4, manifest, faiss)
    """
    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    status = processing_status[job_id]

    if status["stage"] != ProcessingStage.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Processing not completed yet"
        )

    if not status["files"] or file_type not in status["files"]:
        raise HTTPException(
            status_code=404,
            detail=f"File type '{file_type}' not available"
        )

    file_path = Path(status["files"][file_type])

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found on disk"
        )

    # Return file info (in a real implementation, you'd stream the file)
    return {
        "file_path": str(file_path),
        "file_size": file_path.stat().st_size,
        "available_types": list(status["files"].keys())
    }