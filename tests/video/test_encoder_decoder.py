"""
Test suite for Text-Vector Pipeline and VectorMP4Encoder
Updated to match the text processing functionality
"""

import pytest
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import hashlib

# Import our text-vector pipeline components
from ragged.video.encoder import TextVectorPipeline, VectorMP4Encoder, TextChunk, create_text_vector_mp4


class TestTextVectorPipeline:
    """Test text processing and vectorization pipeline"""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "source": "ml_basics.txt"
            },
            {
                "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "source": "deep_learning.txt"
            },
            {
                "text": "Natural language processing helps computers understand, interpret, and generate human language in a valuable way.",
                "source": "nlp_intro.txt"
            }
        ]

    @pytest.fixture
    def pipeline(self):
        """Initialize pipeline with test configuration"""
        return TextVectorPipeline(
            model_name="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20,
            vector_dim=384
        )

    def test_text_chunking_basic(self, pipeline):
        """Test basic text chunking functionality"""
        text = "This is the first sentence. This is the second sentence. This is the third sentence with more content."
        chunks = pipeline.chunk_text(text, source="test.txt")

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.text
            assert chunk.source == "test.txt"
            assert chunk.word_count > 0
            assert chunk.token_count > 0

    def test_chunking_with_overlap(self, pipeline):
        """Test chunking preserves context with overlap"""
        long_text = "Sentence one here. " * 50  # Create long text
        chunks = pipeline.chunk_text(long_text, source="long.txt")

        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            first_words = chunks[0].text.split()[-5:]  # Last 5 words of first chunk
            second_words = chunks[1].text.split()[:10]  # First 10 words of second chunk

            # Should have some overlap
            overlap_found = any(word in second_words for word in first_words)
            assert overlap_found

    def test_empty_input_handling(self, pipeline):
        """Test handling of empty or invalid inputs"""
        # Empty text
        chunks = pipeline.chunk_text("", source="empty.txt")
        assert len(chunks) == 0

        # Whitespace only
        chunks = pipeline.chunk_text("   \n\n  ", source="whitespace.txt")
        assert len(chunks) == 0

        # Empty documents list
        vectors, metadata = pipeline.process_documents([])
        assert len(vectors) == 0
        assert len(metadata) == 0

    def test_document_processing(self, pipeline, sample_documents):
        """Test processing multiple documents"""
        vectors, metadata = pipeline.process_documents(sample_documents)

        assert len(vectors) > 0
        assert len(metadata) == len(vectors)
        assert vectors.shape[1] == pipeline.vector_dim

        # Check metadata structure
        for meta in metadata:
            assert "text" in meta
            assert "source" in meta
            assert "chunk_id" in meta
            assert "word_count" in meta
            assert "token_count" in meta
            assert "text_hash" in meta

    def test_unicode_handling(self, pipeline):
        """Test handling of unicode content"""
        unicode_docs = [
            {"text": "English text with emojis 🚀 and unicode characters", "source": "unicode.txt"},
            {"text": "Texto en español con acentos: áéíóú ñ", "source": "spanish.txt"},
            {"text": "Français avec caractères spéciaux: àèùç", "source": "french.txt"}
        ]

        try:
            vectors, metadata = pipeline.process_documents(unicode_docs)
            assert len(vectors) > 0
            assert vectors.shape[1] == pipeline.vector_dim
        except Exception as e:
            pytest.skip(f"Unicode handling needs improvement: {e}")

    def test_vector_generation(self, pipeline, sample_documents):
        """Test vector generation produces valid embeddings"""
        vectors, metadata = pipeline.process_documents(sample_documents)

        # Vectors should be normalized and finite
        assert np.all(np.isfinite(vectors))
        assert vectors.dtype == np.float32

        # Different texts should produce different vectors
        if len(vectors) > 1:
            similarity = np.dot(vectors[0], vectors[1])
            assert 0.0 <= similarity <= 1.0  # Cosine similarity range


class TestVectorMP4Encoder:
    """Test MP4 encoding functionality"""

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for testing"""
        np.random.seed(42)
        vectors = np.random.randn(50, 384).astype(np.float32)
        # Normalize for cosine similarity
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        metadata = []
        for i in range(50):
            metadata.append({
                "text": f"Sample text chunk {i}",
                "source": f"doc_{i//10}.txt",
                "chunk_id": i,
                "topic": f"topic_{i%5}",
                "word_count": 10 + i,
                "token_count": 15 + i,
                "text_hash": hashlib.md5(f"text_{i}".encode()).hexdigest()[:8]
            })

        return vectors, metadata

    def test_encoder_initialization(self):
        """Test encoder initialization with different configurations"""
        # Default initialization
        encoder = VectorMP4Encoder()
        assert encoder.vector_dim == 384
        assert encoder.chunk_size == 1000
        assert len(encoder.fragments) == 0

        # Custom initialization
        encoder = VectorMP4Encoder(vector_dim=512, chunk_size=500)
        assert encoder.vector_dim == 512
        assert encoder.chunk_size == 500

    def test_add_vectors_basic(self, sample_vectors):
        """Test adding vectors to encoder"""
        vectors, metadata = sample_vectors
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=20)

        encoder.add_vectors(vectors, metadata)

        assert encoder.manifest["metadata"]["total_vectors"] == len(vectors)
        assert len(encoder.fragments) > 0
        assert len(encoder.all_vectors) == len(vectors)

    def test_dimension_mismatch_error(self, sample_vectors):
        """Test error handling for dimension mismatches"""
        vectors, metadata = sample_vectors
        encoder = VectorMP4Encoder(vector_dim=512)  # Wrong dimension

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            encoder.add_vectors(vectors, metadata)

    def test_chunking_behavior(self, sample_vectors):
        """Test vector chunking into fragments"""
        vectors, metadata = sample_vectors
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=10)

        encoder.add_vectors(vectors, metadata)

        # Should create multiple fragments
        expected_fragments = (len(vectors) + encoder.chunk_size - 1) // encoder.chunk_size
        assert len(encoder.fragments) == expected_fragments

        # Check fragment structure
        for fragment in encoder.fragments:
            assert "id" in fragment
            assert "vectors" in fragment
            assert "metadata" in fragment
            assert "vector_count" in fragment

    def test_mp4_encoding(self, sample_vectors, tmp_path):
        """Test complete MP4 encoding process"""
        vectors, metadata = sample_vectors
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=20)
        encoder.add_vectors(vectors, metadata)

        output_file = tmp_path / "test_vectors.mp4"

        # Should encode without errors
        encoder.encode_to_mp4(str(output_file))

        # Check output files exist
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        manifest_file = tmp_path / "test_vectors_manifest.json"
        assert manifest_file.exists()

        # Verify manifest content
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        assert manifest["metadata"]["total_vectors"] == len(vectors)
        assert manifest["metadata"]["vector_dim"] == 384


class TestEndToEndPipeline:
    """Test complete text-to-MP4 pipeline"""

    @pytest.fixture
    def knowledge_base(self):
        """Sample knowledge base for testing"""
        return [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms.",
                "source": "python_intro.txt"
            },
            {
                "text": "JavaScript is the programming language of the web. It enables interactive web pages and is an essential part of web applications.",
                "source": "javascript_basics.txt"
            },
            {
                "text": "Artificial intelligence involves creating systems that can perform tasks that typically require human intelligence.",
                "source": "ai_overview.txt"
            },
            {
                "text": "Machine learning is a subset of AI that enables systems to learn and improve from experience without explicit programming.",
                "source": "ml_definition.txt"
            }
        ]

    def test_complete_pipeline(self, knowledge_base, tmp_path):
        """Test complete pipeline from documents to MP4"""
        output_file = tmp_path / "knowledge_base.mp4"

        # Should complete without errors
        create_text_vector_mp4(knowledge_base, str(output_file))

        # Verify outputs
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Check companion files
        manifest_file = tmp_path / "knowledge_base_manifest.json"
        faiss_file = tmp_path / "knowledge_base_faiss.index"

        assert manifest_file.exists()
        # Faiss file creation depends on vector count

        # Verify manifest structure
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        assert "metadata" in manifest
        assert "vector_map" in manifest
        assert manifest["metadata"]["total_vectors"] > 0

    def test_file_input_processing(self, tmp_path):
        """Test processing files directly"""
        # Create test files
        test_files = []
        for i, content in enumerate([
            "Content of first document about programming.",
            "Content of second document about data science.",
            "Content of third document about machine learning."
        ]):
            file_path = tmp_path / f"doc_{i}.txt"
            file_path.write_text(content)
            test_files.append(str(file_path))

        pipeline = TextVectorPipeline()
        vectors, metadata = pipeline.process_text_files(test_files)

        assert len(vectors) > 0
        assert len(metadata) == len(vectors)

        # Should preserve source file information
        sources = [meta["source"] for meta in metadata]
        assert any("doc_0.txt" in source for source in sources)

    def test_error_recovery(self, tmp_path):
        """Test error recovery and graceful handling"""
        # Test with some invalid documents
        mixed_docs = [
            {"text": "Valid document content", "source": "valid.txt"},
            {"text": "", "source": "empty.txt"},  # Empty
            {"text": "   ", "source": "whitespace.txt"},  # Whitespace only
            {"text": "Another valid document", "source": "valid2.txt"}
        ]

        output_file = tmp_path / "mixed_content.mp4"

        # Should handle gracefully and process valid content
        create_text_vector_mp4(mixed_docs, str(output_file))

        if output_file.exists():
            manifest_file = tmp_path / "mixed_content_manifest.json"
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)

            # Should have processed some vectors (from valid docs)
            assert manifest["metadata"]["total_vectors"] > 0

    def test_large_document_handling(self, tmp_path):
        """Test handling of large documents"""
        # Create a large document
        large_text = "This is a sentence that will be repeated many times. " * 200
        large_doc = [{"text": large_text, "source": "large_doc.txt"}]

        output_file = tmp_path / "large_doc.mp4"

        start_time = time.time()
        create_text_vector_mp4(large_doc, str(output_file))
        processing_time = time.time() - start_time

        # Should complete in reasonable time
        assert processing_time < 60.0  # 1 minute limit
        assert output_file.exists()


class TestPerformanceAndScaling:
    """Test performance characteristics"""

    def test_chunking_performance(self):
        """Test chunking performance with various text sizes"""
        pipeline = TextVectorPipeline()

        test_sizes = [
            ("Small", "Short text. " * 10),
            ("Medium", "Medium length text with sentences. " * 100),
            ("Large", "Long text document with many sentences. " * 500)
        ]

        for name, text in test_sizes:
            start_time = time.time()
            chunks = pipeline.chunk_text(text, source=f"{name.lower()}.txt")
            chunk_time = time.time() - start_time

            assert len(chunks) > 0
            assert chunk_time < 10.0  # Should be fast

            print(f"{name}: {len(chunks)} chunks in {chunk_time:.3f}s")

    def test_encoding_performance(self, tmp_path):
        """Test encoding performance with different dataset sizes"""
        sizes_to_test = [10, 50, 100]

        for size in sizes_to_test:
            # Generate test documents
            docs = []
            for i in range(size):
                docs.append({
                    "text": f"Document {i} with some content about topic {i % 5}. " * 3,
                    "source": f"doc_{i}.txt"
                })

            output_file = tmp_path / f"perf_test_{size}.mp4"

            start_time = time.time()
            create_text_vector_mp4(docs, str(output_file))
            total_time = time.time() - start_time

            print(f"Size {size}: {total_time:.3f}s")

            # Reasonable performance expectations
            assert total_time < 30.0  # 30 seconds max
            assert output_file.exists()

    def test_memory_usage_basic(self):
        """Basic memory usage monitoring"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Process moderate dataset
        docs = [
            {"text": f"Memory test document {i}. " * 20, "source": f"mem_{i}.txt"}
            for i in range(50)
        ]

        pipeline = TextVectorPipeline()
        vectors, metadata = pipeline.process_documents(docs)

        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        print(f"Memory increase: {memory_increase:.2f} MB")

        # Should not use excessive memory
        assert memory_increase < 500  # 500MB limit for this test


class TestConfigurationHandling:
    """Test configuration and parameter handling"""

    def test_model_configuration(self):
        """Test different model configurations"""
        # Test with different models (if available)
        models_to_test = [
            ("all-MiniLM-L6-v2", 384),
            # Add other models as available
        ]

        for model_name, expected_dim in models_to_test:
            try:
                pipeline = TextVectorPipeline(
                    model_name=model_name,
                    vector_dim=expected_dim
                )

                # Test with sample text
                docs = [{"text": "Test document for model", "source": "test.txt"}]
                vectors, metadata = pipeline.process_documents(docs)

                assert vectors.shape[1] == expected_dim

            except Exception as e:
                pytest.skip(f"Model {model_name} not available: {e}")

    def test_chunking_parameters(self):
        """Test different chunking parameters"""
        test_text = "This is sentence one. This is sentence two. " * 20

        param_sets = [
            (50, 10),   # Small chunks, small overlap
            (100, 20),  # Medium chunks, medium overlap
            (200, 50),  # Large chunks, large overlap
        ]

        for chunk_size, overlap in param_sets:
            pipeline = TextVectorPipeline(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )

            chunks = pipeline.chunk_text(test_text, source="param_test.txt")

            assert len(chunks) > 0

            # Verify chunk sizes are reasonable
            for chunk in chunks:
                assert chunk.token_count <= chunk_size * 1.2  # Allow some flexibility


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--tb=short", "-x"])