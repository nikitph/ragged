"""
Test suite for PDF upload endpoint with progress tracking
"""

import time
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Import the endpoint
from ragged.api.v1.endpoints.pdf_upload import (
    router as pdf_router,
    ProcessingStage,
    extract_text_from_pdf,
    processing_status
)


class TestPDFUploadEndpoint:
    """Test PDF upload functionality"""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with PDF upload router"""
        app = FastAPI()
        app.include_router(pdf_router, prefix="/api/v1/videos")
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_pdf_bytes(self):
        """Create a sample PDF for testing"""
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        # Add some text content
        p.drawString(100, 750, "Test PDF Document")
        p.drawString(100, 700, "This is a sample PDF for testing the upload endpoint.")
        p.drawString(100, 650, "It contains multiple lines of text.")
        p.drawString(100, 600, "Machine learning and artificial intelligence are important topics.")
        p.drawString(100, 550, "Deep learning uses neural networks for pattern recognition.")

        p.showPage()
        p.save()

        buffer.seek(0)
        return buffer.getvalue()

    @pytest.fixture
    def empty_pdf_bytes(self):
        """Create an empty PDF for testing"""
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.showPage()  # Empty page
        p.save()

        buffer.seek(0)
        return buffer.getvalue()

    @pytest.fixture
    def large_pdf_bytes(self):
        """Create a large PDF for testing size limits"""
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        # Add lots of text to make it large
        for page in range(10):
            for line in range(50):
                p.drawString(50, 750 - line * 15, f"Page {page} Line {line}: " + "Large content " * 20)
            p.showPage()

        p.save()
        buffer.seek(0)
        return buffer.getvalue()

    def test_extract_text_from_pdf(self, sample_pdf_bytes):
        """Test PDF text extraction"""
        text = extract_text_from_pdf(sample_pdf_bytes)

        assert len(text) > 0
        assert "Test PDF Document" in text
        assert "Machine learning" in text
        assert "neural networks" in text

    def test_extract_text_from_empty_pdf(self, empty_pdf_bytes):
        """Test extraction from empty PDF"""
        text = extract_text_from_pdf(empty_pdf_bytes)

        # Should return empty string or minimal content
        assert isinstance(text, str)

    def test_extract_text_from_invalid_pdf(self):
        """Test extraction from invalid PDF data"""
        invalid_data = b"This is not a PDF file"

        with pytest.raises(ValueError, match="Failed to extract text from PDF"):
            extract_text_from_pdf(invalid_data)

    def test_pdf_upload_success(self, client, sample_pdf_bytes):
        """Test successful PDF upload"""
        files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}

        response = client.post("/api/v1/videos/upload-pdf", files=files)

        assert response.status_code == 200
        data = response.json()

        assert "job_id" in data
        assert "message" in data
        assert "status_url" in data
        assert "test.pdf" in data["message"]

    def test_pdf_upload_wrong_file_type(self, client):
        """Test upload with non-PDF file"""
        files = {"file": ("test.txt", BytesIO(b"Not a PDF"), "text/plain")}

        response = client.post("/api/v1/videos/upload-pdf", files=files)

        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["detail"]

    def test_pdf_upload_empty_file(self, client):
        """Test upload with empty file"""
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}

        response = client.post("/api/v1/videos/upload-pdf", files=files)

        assert response.status_code == 400
        assert "Empty file received" in response.json()["detail"]

    @patch('ragged.api.v1.endpoints.pdf_upload.max_size', 1024)  # Mock small size limit
    def test_pdf_upload_file_too_large(self, client, large_pdf_bytes):
        """Test upload with file exceeding size limit"""
        files = {"file": ("large.pdf", BytesIO(large_pdf_bytes), "application/pdf")}

        response = client.post("/api/v1/videos/upload-pdf", files=files)

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_get_processing_status_not_found(self, client):
        """Test getting status for non-existent job"""
        response = client.get("/api/v1/videos/status/nonexistent-job-id")

        assert response.status_code == 404
        assert "Job ID not found" in response.json()["detail"]

    def test_processing_status_flow(self, client, sample_pdf_bytes):
        """Test the complete processing status flow"""
        # Clear any existing status
        processing_status.clear()

        # Upload PDF
        files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
        upload_response = client.post("/api/v1/videos/upload-pdf", files=files)

        assert upload_response.status_code == 200
        job_id = upload_response.json()["job_id"]

        # Check initial status
        status_response = client.get(f"/api/v1/videos/status/{job_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["stage"] in [ProcessingStage.UPLOADING, ProcessingStage.EXTRACTING_TEXT]
        assert 0.0 <= status_data["progress"] <= 1.0

    def test_list_processing_jobs(self, client, sample_pdf_bytes):
        """Test listing all processing jobs"""
        # Clear existing jobs
        processing_status.clear()

        # Upload a PDF to create a job
        files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
        upload_response = client.post("/api/v1/videos/upload-pdf", files=files)
        job_id = upload_response.json()["job_id"]

        # List jobs
        response = client.get("/api/v1/videos/list-jobs")

        assert response.status_code == 200
        data = response.json()

        assert job_id in data["jobs"]
        assert data["total"] >= 1

    def test_cleanup_job_status(self, client, sample_pdf_bytes):
        """Test cleaning up job status"""
        # Upload PDF to create a job
        files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
        upload_response = client.post("/api/v1/videos/upload-pdf", files=files)
        job_id = upload_response.json()["job_id"]

        # Verify job exists
        status_response = client.get(f"/api/v1/videos/status/{job_id}")
        assert status_response.status_code == 200

        # Clean up job
        cleanup_response = client.delete(f"/api/v1/videos/status/{job_id}")
        assert cleanup_response.status_code == 200

        # Verify job is gone
        status_response = client.get(f"/api/v1/videos/status/{job_id}")
        assert status_response.status_code == 404

    def test_download_file_not_completed(self, client, sample_pdf_bytes):
        """Test downloading file when processing not completed"""
        # Upload PDF
        files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
        upload_response = client.post("/api/v1/videos/upload-pdf", files=files)
        job_id = upload_response.json()["job_id"]

        # Try to download before completion
        response = client.get(f"/api/v1/videos/download/{job_id}/mp4")

        assert response.status_code == 400
        assert "Processing not completed" in response.json()["detail"]

    def test_download_file_not_found_job(self, client):
        """Test downloading file for non-existent job"""
        response = client.get("/api/v1/videos/download/nonexistent/mp4")

        assert response.status_code == 404
        assert "Job ID not found" in response.json()["detail"]


class TestProcessingStages:
    """Test the processing stages functionality"""

    @pytest.fixture
    def mock_encoder(self):
        """Mock the video encoder components"""
        with patch('ragged.api.v1.endpoints.pdf_upload.TextVectorPipeline') as mock_pipeline, \
                patch('ragged.api.v1.endpoints.pdf_upload.VectorMP4Encoder') as mock_encoder:
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.process_documents.return_value = (
                [[0.1, 0.2, 0.3]] * 10,  # Mock vectors
                [{"text": f"chunk {i}", "source": "test.pdf"} for i in range(10)]  # Mock metadata
            )
            mock_pipeline.return_value = mock_pipeline_instance

            # Mock encoder
            mock_encoder_instance = Mock()
            mock_encoder_instance.add_vectors = Mock()
            mock_encoder_instance.encode_to_mp4 = Mock()
            mock_encoder.return_value = mock_encoder_instance

            yield mock_pipeline, mock_encoder

    @pytest.mark.asyncio
    async def test_processing_stages_progression(self, mock_encoder, sample_pdf_bytes):
        """Test that processing stages progress correctly"""
        from ragged.api.v1.endpoints.pdf_upload import process_pdf_to_video

        job_id = "test-job-123"
        processing_status[job_id] = {
            "job_id": job_id,
            "stage": ProcessingStage.UPLOADING,
            "progress": 0.0,
            "message": "Starting...",
            "error": None,
            "files": None,
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None
        }

        # Mock file system operations
        with patch('pathlib.Path.mkdir'), \
                patch('pathlib.Path.exists', return_value=True):
            await process_pdf_to_video(job_id, sample_pdf_bytes, "test.pdf")

            # Check final status
            final_status = processing_status[job_id]
            assert final_status["stage"] == ProcessingStage.COMPLETED
            assert final_status["progress"] == 1.0
            assert final_status["files"] is not None

    @pytest.mark.asyncio
    async def test_processing_error_handling(self, sample_pdf_bytes):
        """Test error handling during processing"""
        from ragged.api.v1.endpoints.pdf_upload import process_pdf_to_video

        job_id = "test-error-job"
        processing_status[job_id] = {
            "job_id": job_id,
            "stage": ProcessingStage.UPLOADING,
            "progress": 0.0,
            "message": "Starting...",
            "error": None,
            "files": None,
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None
        }

        # Mock an error in the pipeline
        with patch('ragged.api.v1.endpoints.pdf_upload.TextVectorPipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Mock processing error")

            await process_pdf_to_video(job_id, sample_pdf_bytes, "test.pdf")

            # Check error status
            final_status = processing_status[job_id]
            assert final_status["stage"] == ProcessingStage.FAILED
            assert final_status["error"] is not None
            assert "Mock processing error" in final_status["error"]

    @pytest.mark.asyncio
    async def test_empty_pdf_processing(self, empty_pdf_bytes):
        """Test processing PDF with no extractable text"""
        from ragged.api.v1.endpoints.pdf_upload import process_pdf_to_video

        job_id = "test-empty-job"
        processing_status[job_id] = {
            "job_id": job_id,
            "stage": ProcessingStage.UPLOADING,
            "progress": 0.0,
            "message": "Starting...",
            "error": None,
            "files": None,
            "started_at": "2024-01-01T00:00:00",
            "completed_at": None
        }

        await process_pdf_to_video(job_id, empty_pdf_bytes, "empty.pdf")

        # Should fail due to no text content
        final_status = processing_status[job_id]
        assert final_status["stage"] == ProcessingStage.FAILED
        assert "No text content could be extracted" in final_status["error"]


class TestIntegration:
    """Integration tests with real components"""

    @pytest.fixture
    def storage_dir(self, tmp_path):
        """Mock storage directory"""
        storage_dir = tmp_path / "storage" / "uploads"
        storage_dir.mkdir(parents=True)

        with patch('ragged.api.v1.endpoints.pdf_upload.get_storage_dir', return_value=storage_dir):
            yield storage_dir

    def test_end_to_end_with_mocked_encoder(self, client, sample_pdf_bytes, storage_dir):
        """Test end-to-end flow with mocked video encoder"""

        # Mock the encoder components to avoid actual processing
        with patch('ragged.api.v1.endpoints.pdf_upload.TextVectorPipeline') as mock_pipeline, \
                patch('ragged.api.v1.endpoints.pdf_upload.VectorMP4Encoder') as mock_encoder:

            # Setup mocks
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.process_documents.return_value = (
                [[0.1] * 384] * 5,  # 5 vectors of 384 dimensions
                [{"text": f"chunk {i}", "source": "test.pdf"} for i in range(5)]
            )
            mock_pipeline.return_value = mock_pipeline_instance

            mock_encoder_instance = Mock()
            mock_encoder.return_value = mock_encoder_instance

            # Mock file creation
            def mock_encode_to_mp4(path):
                # Create mock files
                Path(path).touch()
                manifest_path = path.replace('.mp4', '_manifest.json')
                faiss_path = path.replace('.mp4', '_faiss.index')
                Path(manifest_path).touch()
                Path(faiss_path).touch()

            mock_encoder_instance.encode_to_mp4.side_effect = mock_encode_to_mp4

            # Upload PDF
            files = {"file": ("test.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
            upload_response = client.post("/api/v1/videos/upload-pdf", files=files)

            assert upload_response.status_code == 200
            job_id = upload_response.json()["job_id"]

            # Wait a moment for background processing
            time.sleep(0.1)

            # Check status multiple times to see progression
            max_attempts = 10
            for attempt in range(max_attempts):
                status_response = client.get(f"/api/v1/videos/status/{job_id}")
                status_data = status_response.json()

                if status_data["stage"] == ProcessingStage.COMPLETED:
                    assert status_data["files"] is not None
                    assert "mp4" in status_data["files"]
                    break

                time.sleep(0.1)
            else:
                pytest.fail("Processing did not complete within expected time")

    def test_concurrent_uploads(self, client, sample_pdf_bytes):
        """Test handling multiple concurrent uploads"""

        # Mock encoder to make processing fast
        with patch('ragged.api.v1.endpoints.pdf_upload.TextVectorPipeline'), \
                patch('ragged.api.v1.endpoints.pdf_upload.VectorMP4Encoder'):

            # Upload multiple PDFs concurrently
            job_ids = []
            for i in range(3):
                files = {"file": (f"test_{i}.pdf", BytesIO(sample_pdf_bytes), "application/pdf")}
                response = client.post("/api/v1/videos/upload-pdf", files=files)
                assert response.status_code == 200
                job_ids.append(response.json()["job_id"])

            # Verify all jobs are tracked
            jobs_response = client.get("/api/v1/videos/list-jobs")
            jobs_data = jobs_response.json()

            for job_id in job_ids:
                assert job_id in jobs_data["jobs"]


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])