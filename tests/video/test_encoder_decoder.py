"""
Test suite for Ragged video encoding and decoding functionality
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from ragged.video import VideoEncoder, VideoRetriever
from ragged.video.utils import chunk_text, encode_to_qr, decode_qr, qr_to_frame


class TestVideoProcessing:
    """Test video encoding and retrieval pipeline"""

    @pytest.fixture
    def test_chunks(self):
        """Sample chunks for testing"""
        return [
            "Artificial intelligence is transforming how we work and live.",
            "Machine learning algorithms can process vast amounts of data.",
            "Deep learning neural networks are inspired by brain structure.",
            "Natural language processing enables human-computer communication.",
            "Computer vision allows machines to interpret visual information."
        ]

    def test_video_encoder_basic(self, test_chunks, temp_dir):
        """Test basic video encoding functionality"""
        encoder = VideoEncoder()
        encoder.add_chunks(test_chunks)

        # Check encoder stats
        stats = encoder.get_stats()
        assert stats['total_chunks'] == len(test_chunks)
        assert stats['total_characters'] > 0
        assert stats['avg_chunk_size'] > 0

        # Build video
        video_file = temp_dir / "test.mp4"
        index_file = temp_dir / "test_index.json"

        build_stats = encoder.build_video(
            str(video_file),
            str(index_file),
            codec="mp4v",
            show_progress=False
        )

        # Verify outputs
        assert video_file.exists()
        assert index_file.exists()
        assert (temp_dir / "test_index.faiss").exists()

        # Check build stats
        assert build_stats['backend'] == 'opencv'
        assert build_stats['total_frames'] == len(test_chunks)
        assert build_stats['video_size_mb'] > 0

    def test_video_retriever_basic(self, test_chunks, temp_dir):
        """Test basic video retrieval functionality"""
        # First create a video
        encoder = VideoEncoder()
        encoder.add_chunks(test_chunks)

        video_file = temp_dir / "test.mp4"
        index_file = temp_dir / "test_index.json"

        encoder.build_video(str(video_file), str(index_file), codec="mp4v", show_progress=False)

        # Now test retrieval
        retriever = VideoRetriever(str(video_file), str(index_file))

        # Test search
        results = retriever.search("artificial intelligence", top_k=3)
        assert len(results) <= 3
        assert len(results) > 0
        assert all(isinstance(result, str) for result in results)

        # Test search with metadata
        detailed_results = retriever.search_with_metadata("machine learning", top_k=2)
        assert len(detailed_results) <= 2

        for result in detailed_results:
            assert 'text' in result
            assert 'score' in result
            assert 'chunk_id' in result
            assert 'frame' in result
            assert 0 <= result['score'] <= 1

    def test_chunk_by_id_retrieval(self, test_chunks, temp_dir):
        """Test retrieving specific chunks by ID"""
        encoder = VideoEncoder()
        encoder.add_chunks(test_chunks)

        video_file = temp_dir / "test.mp4"
        index_file = temp_dir / "test_index.json"

        encoder.build_video(str(video_file), str(index_file), codec="mp4v", show_progress=False)

        retriever = VideoRetriever(str(video_file), str(index_file))

        # Test retrieving first chunk
        chunk_0 = retriever.get_chunk_by_id(0)
        assert chunk_0 is not None
        assert isinstance(chunk_0, str)

        # Test invalid chunk ID
        invalid_chunk = retriever.get_chunk_by_id(999)
        assert invalid_chunk is None


class TestTextProcessing:
    """Test text processing utilities"""

    def test_text_chunking(self):
        """Test text chunking functionality"""
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        assert all(len(chunk) <= 35 for chunk in chunks)  # Allow for sentence boundaries
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks

    def test_qr_encoding_decoding(self):
        """Test QR code generation and decoding"""
        test_data = "Hello from Ragged QR test!"

        # Encode to QR
        qr_image = encode_to_qr(test_data)
        assert qr_image is not None

        # Convert to frame
        frame = qr_to_frame(qr_image, (256, 256))
        assert frame.shape == (256, 256, 3)

        # Decode
        decoded = decode_qr(frame)
        assert decoded == test_data

    def test_qr_compression(self):
        """Test QR compression for large data"""
        # Large data that should trigger compression
        large_data = "x" * 500

        qr_image = encode_to_qr(large_data)
        frame = qr_to_frame(qr_image, (512, 512))
        decoded = decode_qr(frame)

        assert decoded == large_data


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_end_to_end_workflow(self, temp_dir):
        """Test complete encode → search → retrieve workflow"""
        # Sample knowledge base
        knowledge = [
            "Python is a high-level programming language known for its simplicity.",
            "JavaScript is widely used for web development and frontend applications.",
            "Machine learning models require large datasets for effective training.",
            "Cloud computing provides scalable infrastructure for modern applications.",
            "Database optimization is crucial for application performance."
        ]

        # Encode
        encoder = VideoEncoder()
        encoder.add_chunks(knowledge)

        video_file = temp_dir / "knowledge.mp4"
        index_file = temp_dir / "knowledge_index.json"

        build_stats = encoder.build_video(
            str(video_file), str(index_file),
            codec="mp4v", show_progress=False
        )

        assert build_stats['total_chunks'] == len(knowledge)

        # Retrieve and search
        retriever = VideoRetriever(str(video_file), str(index_file))

        # Test various queries
        python_results = retriever.search("Python programming", top_k=2)
        assert len(python_results) > 0
        assert any("Python" in result for result in python_results)

        web_results = retriever.search("web development", top_k=2)
        assert len(web_results) > 0
        assert any("JavaScript" in result or "web" in result for result in web_results)

        # Test integrity
        integrity = retriever.validate_integrity()
        assert integrity['is_healthy']
        assert integrity['integrity_percent'] >= 95

    def test_text_file_processing(self, temp_dir):
        """Test processing text from files"""
        # Create a test text file
        test_file = temp_dir / "test_document.txt"
        test_content = """
        Artificial Intelligence and Machine Learning

        Artificial intelligence (AI) represents one of the most significant technological 
        advances of our time. Machine learning, a subset of AI, enables computers to 
        learn and improve from experience without being explicitly programmed.

        Deep learning neural networks are inspired by the human brain and can recognize
        complex patterns in vast datasets. Natural language processing allows computers
        to understand and generate human language.
        """

        test_file.write_text(test_content)

        # Process file
        encoder = VideoEncoder()
        encoder.add_file(str(test_file), chunk_size=200, overlap=30)

        stats = encoder.get_stats()
        assert stats['total_chunks'] > 1
        assert stats['total_characters'] > len(test_content) * 0.8  # Account for chunking

        # Build and test
        video_file = temp_dir / "document.mp4"
        index_file = temp_dir / "document_index.json"

        encoder.build_video(str(video_file), str(index_file), codec="mp4v", show_progress=False)

        retriever = VideoRetriever(str(video_file), str(index_file))
        results = retriever.search("artificial intelligence", top_k=3)

        assert len(results) > 0
        assert any("artificial" in result.lower() for result in results)


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])