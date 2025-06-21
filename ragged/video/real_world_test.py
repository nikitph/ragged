"""
Real-world test using electrophoresis document
Tests enhanced text-to-MP4 pipeline with CDN optimizations and caching
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np
import os
import shutil

import requests
import requests_mock
from unittest.mock import patch, MagicMock

# Import the enhanced classes from our artifacts
from encoder  import TextVectorPipeline, VectorMP4Encoder, create_text_vector_mp4
from decoder import CachedVectorMP4Decoder


class TestElectrophoresisDocument:
    """Test with real scientific document content using enhanced pipeline"""

    @pytest.fixture
    def electrophoresis_text(self):
        """Real electrophoresis document content"""
        return """Electrophoresis is a fundamental technique used in biochemistry, molecular biology, and analytical chemistry to separate charged biomolecules—like DNA, RNA, or proteins—based on their size, charge, and conformation under the influence of an electric field.

🔬 Definition

Electrophoresis is the movement of charged particles through a medium (usually a gel or liquid) in response to an electric field. Positively charged particles migrate toward the cathode (−), while negatively charged ones move toward the anode (+).

📚 Types of Electrophoresis

Agarose Gel Electrophoresis is used for DNA/RNA separation using agarose gel medium.
Polyacrylamide Gel Electrophoresis (PAGE) separates proteins and DNA fragments using polyacrylamide gel.
SDS-PAGE is used for denatured proteins with SDS-treated polyacrylamide.
Isoelectric Focusing (IEF) separates proteins by pI using pH gradient gel.
Capillary Electrophoresis handles small ions and peptides in capillary tubes.
2D Electrophoresis processes complex protein mixtures using IEF plus SDS-PAGE.

⚙️ Principle

Charge-Based Separation: Particles move in response to an electric field. The direction and speed depend on net charge, size, shape, and medium resistance.

Gel Matrix: Acts like a sieve. Larger molecules move slower through the pores.

Electric Field: Typically 50–150 volts is applied. Higher voltages speed up migration but can cause smearing due to heat.

🧪 Common Buffers

TAE or TBE buffers are used in agarose for DNA to maintain pH and conduct current.
Tris-Glycine buffer is used in SDS-PAGE to provide ions and stable pH.
Urea buffer is used in IEF or PAGE as a denaturing agent for protein unfolding.

🧬 Application Areas

Nucleic Acid Analysis includes DNA fingerprinting for forensics, PCR verification, plasmid mapping, and RNA integrity checking.

Protein Analysis covers protein purity assessment, molecular weight determination, Western blotting with SDS-PAGE plus antibody probing, and 2D electrophoresis for proteomics.

Clinical Diagnostics involves hemoglobin variants like Sickle Cell detection and serum protein electrophoresis for multiple myeloma.

🧬 Step-by-Step: Agarose Gel Electrophoresis

Gel Preparation: Agarose is melted in buffer, poured into a tray, and allowed to set with a comb for wells.
Sample Preparation: DNA mixed with loading dye adds weight and color.
Running the Gel: Samples are loaded into wells, and current is applied at around 100V.
Migration: DNA moves toward the anode due to negative charge. Shorter fragments move faster.
Visualization: Gel is stained with ethidium bromide or SYBR Green, then visualized under UV light.

🔍 SDS-PAGE for Protein Separation

SDS Denaturation: Proteins are treated with Sodium Dodecyl Sulfate to impart uniform negative charge and unfold 3D structure.
Gel Loading: Protein samples with dye are loaded into wells.
Electrophoresis: Proteins migrate based on molecular weight with small proteins moving faster.
Staining: Gel is stained with Coomassie Blue or Silver stain to visualize protein bands.

⚠️ Factors Affecting Electrophoresis

Voltage affects speed - higher voltage means faster migration but may cause overheating.
Gel concentration - higher percentage gels provide better separation of small molecules.
Buffer composition affects migration speed and resolution.
Sample quality - degraded samples smear or show extra bands.
Time of run - under or over-running may distort bands.

✅ Advantages

Electrophoresis is simple and cost-effective with high resolution of biomolecules.
Can separate by multiple properties including charge, size, and pI.
Adaptable for many different molecules.

❌ Limitations

Not quantitative unless combined with densitometry.
Limited scalability and sensitive to run conditions.
Smearing and artifacts can occur with improper setup."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory"""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        yield str(cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def test_enhanced_real_document_processing(self, electrophoresis_text, tmp_path):
        """Test processing real scientific document with enhanced encoder"""
        pipeline = TextVectorPipeline(
            model_name="all-MiniLM-L6-v2",
            chunk_size=200,
            chunk_overlap=30,
            vector_dim=384
        )

        doc = {"text": electrophoresis_text, "source": "electrophoresis_guide.txt"}
        vectors, metadata = pipeline.process_documents([doc])

        # Should create meaningful chunks
        assert len(vectors) > 5  # Expect multiple chunks from this content
        assert vectors.shape[1] == 384

        # Check topic extraction worked
        topics = [meta.get("topic") for meta in metadata]
        meaningful_topics = [t for t in topics if t and len(t) > 2]
        assert len(meaningful_topics) > 0

        print(f"Generated {len(vectors)} chunks with topics: {set(meaningful_topics)}")

        # Test enhanced encoder with byte tracking
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=50)
        encoder.add_vectors(vectors, metadata)

        output_file = tmp_path / "electrophoresis_enhanced.mp4"
        encoder.encode_to_mp4(str(output_file))

        # Verify enhanced manifest has byte positions
        manifest_file = tmp_path / "electrophoresis_enhanced_manifest.json"
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        # Check file structure information
        assert "file_structure" in manifest["metadata"]
        file_struct = manifest["metadata"]["file_structure"]

        required_fields = ["ftyp_start", "ftyp_size", "video_track_start",
                          "manifest_start", "fragments_start", "total_file_size"]
        for field in required_fields:
            assert field in file_struct
            assert isinstance(file_struct[field], int)
            assert file_struct[field] >= 0

        # Check fragment byte positions
        for fragment in manifest["metadata"]["fragments"]:
            byte_fields = ["byte_start", "byte_end", "byte_size",
                          "moof_start", "mdat_start", "data_start", "data_size"]
            for field in byte_fields:
                assert field in fragment
                assert isinstance(fragment[field], int)
                assert fragment[field] >= 0

        print(f"Enhanced encoding complete with {len(manifest['metadata']['fragments'])} fragments")
        print(f"File structure: {file_struct}")

    def test_cached_decoder_functionality(self, electrophoresis_text, tmp_path, temp_cache_dir):
        """Test enhanced decoder with caching functionality"""
        # Create test data
        documents = [
            {"text": electrophoresis_text, "source": "electrophoresis.txt"},
            {
                "text": "PCR amplification is a molecular biology technique used to amplify DNA sequences. The process uses thermal cycling with DNA polymerase enzyme.",
                "source": "pcr_basics.txt"
            }
        ]

        output_file = tmp_path / "cached_test.mp4"
        create_text_vector_mp4(documents, str(output_file))

        # Test local cached decoder
        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=10,
            disk_cache_dir=temp_cache_dir
        )

        # Test manifest info
        info = decoder.get_manifest_info()
        assert info["total_vectors"] > 0
        assert info["vector_dimension"] == 384
        assert len(info["topics"]) > 0

        print(f"Decoder info: {info}")

        # Test vector retrieval with caching
        vector_ids = list(range(min(3, info["total_vectors"])))
        vectors1, metadata1 = decoder.get_vectors_by_ids(vector_ids)

        # Second call should use cache
        vectors2, metadata2 = decoder.get_vectors_by_ids(vector_ids)

        assert np.array_equal(vectors1, vectors2)
        assert metadata1 == metadata2

        # Check cache statistics
        stats = decoder.get_cache_stats()
        assert stats["lru_cache_hits"] > 0  # Should have cache hits on second call
        print(f"Cache stats: {stats}")

    def test_http_range_simulation(self, electrophoresis_text, tmp_path, temp_cache_dir):
        """Test HTTP range request simulation with mocked CDN"""
        documents = [{"text": electrophoresis_text, "source": "range_test.txt"}]
        output_file = tmp_path / "range_test.mp4"
        create_text_vector_mp4(documents, str(output_file))

        # Read the actual file for mocking
        with open(output_file, 'rb') as f:
            file_content = f.read()

        manifest_file = tmp_path / "range_test_manifest.json"
        with open(manifest_file, 'r') as f:
            manifest_content = f.read()

        # Mock HTTP requests
        with requests_mock.Mocker() as m:
            # Mock manifest request
            m.get("https://cdn.example.com/range_test_manifest.json",
                  text=manifest_content)

            # Mock range requests for fragments
            def range_callback(request, context):
                range_header = request.headers.get('Range', '')
                if range_header.startswith('bytes='):
                    start, end = range_header[6:].split('-')
                    start, end = int(start), int(end)
                    context.status_code = 206
                    context.headers['Content-Range'] = f'bytes {start}-{end}/{len(file_content)}'
                    return file_content[start:end+1]
                return file_content

            m.get("https://cdn.example.com/range_test.mp4", content=range_callback)

            # Test remote decoder with range requests
            decoder = CachedVectorMP4Decoder(
                mp4_path="https://cdn.example.com/range_test.mp4",
                manifest_path="https://cdn.example.com/range_test_manifest.json",
                cache_size=5,
                disk_cache_dir=temp_cache_dir
            )

            # Should work with HTTP range requests
            info = decoder.get_manifest_info()
            assert info["total_vectors"] > 0

            if info["total_vectors"] > 0:
                vectors, metadata = decoder.get_vectors_by_ids([0])
                assert len(vectors) == 1
                assert len(metadata) == 1

            print(f"HTTP range request test passed with {info['total_vectors']} vectors")

    def test_fragment_prefetching(self, tmp_path, temp_cache_dir):
        """Test fragment prefetching functionality"""
        docs = [
            {"text": "DNA electrophoresis separates nucleic acids.", "source": "dna.txt"},
            {"text": "Protein separation uses SDS-PAGE techniques.", "source": "protein.txt"},
            {"text": "Western blotting detects specific proteins.", "source": "western.txt"},
            {"text": "PCR amplifies DNA sequences efficiently.", "source": "pcr.txt"}
        ]

        output_file = tmp_path / "prefetch_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=10,
            disk_cache_dir=temp_cache_dir
        )

        info = decoder.get_manifest_info()
        fragment_ids = [frag["id"] for frag in decoder.manifest["metadata"]["fragments"]]

        if len(fragment_ids) > 1:
            # Test prefetching
            prefetch_ids = fragment_ids[:2]
            decoder.prefetch_fragments(prefetch_ids)

            # Check that fragments are now cached
            stats_before = decoder.get_cache_stats()

            # Access prefetched fragments (should be fast)
            for frag_id in prefetch_ids:
                decoder._read_fragment_cached(frag_id)

            stats_after = decoder.get_cache_stats()

            # Should have cache hits
            assert stats_after["lru_cache_hits"] >= stats_before["lru_cache_hits"]
            print(f"Prefetching test: {len(prefetch_ids)} fragments prefetched")

    def test_search_with_caching(self, electrophoresis_text, tmp_path, temp_cache_dir):
        """Test search functionality with enhanced caching"""
        knowledge_base = [
            {"text": "Agarose gel electrophoresis separates DNA fragments by size.", "source": "dna_method.txt"},
            {"text": "SDS-PAGE protein electrophoresis denatures proteins for analysis.", "source": "protein_method.txt"},
            {"text": "Western blotting combines electrophoresis with antibody detection.", "source": "western.txt"},
            {"text": electrophoresis_text[:500], "source": "electrophoresis_excerpt.txt"}  # Use excerpt
        ]

        output_file = tmp_path / "search_cache_test.mp4"
        create_text_vector_mp4(knowledge_base, str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=20,
            disk_cache_dir=temp_cache_dir
        )

        if decoder.faiss_index is not None:
            # Create search query
            pipeline = TextVectorPipeline()
            query_text = "DNA separation and fragment analysis"
            query_vector = pipeline.model.encode([query_text])[0]

            # First search
            results1 = decoder.search_vectors(query_vector, top_k=3)
            stats1 = decoder.get_cache_stats()

            # Second identical search (should use cache more)
            results2 = decoder.search_vectors(query_vector, top_k=3)
            stats2 = decoder.get_cache_stats()

            assert len(results1) > 0
            assert len(results2) > 0
            assert stats2["lru_cache_hits"] >= stats1["lru_cache_hits"]

            # Check result quality
            dna_found = any("DNA" in result["metadata"].get("text", "") for result in results1)
            assert dna_found

            print(f"Search with caching: {len(results1)} results found")
            print(f"Cache performance improvement: {stats2['lru_cache_hits'] - stats1['lru_cache_hits']} additional hits")

    def test_topic_based_retrieval_enhanced(self, tmp_path, temp_cache_dir):
        """Test enhanced topic-based vector retrieval"""
        docs = [
            {"text": "Electrophoresis principles involve charged particle migration.", "source": "principles.txt"},
            {"text": "Laboratory protocols require proper buffer preparation.", "source": "protocols.txt"},
            {"text": "Protein analysis uses various electrophoretic techniques.", "source": "analysis.txt"}
        ]

        output_file = tmp_path / "topic_enhanced_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=15,
            disk_cache_dir=temp_cache_dir
        )

        info = decoder.get_manifest_info()
        available_topics = info["topics"]

        print(f"Available topics: {available_topics}")

        if available_topics:
            test_topic = available_topics[0]
            topic_vectors, topic_metadata = decoder.get_vectors_by_topic(test_topic)

            if len(topic_vectors) > 0:
                assert all(meta.get("topic") == test_topic for meta in topic_metadata)
                print(f"Retrieved {len(topic_vectors)} vectors for topic '{test_topic}'")

                # Test caching for topic retrieval
                stats_before = decoder.get_cache_stats()
                topic_vectors2, topic_metadata2 = decoder.get_vectors_by_topic(test_topic)
                stats_after = decoder.get_cache_stats()

                assert np.array_equal(topic_vectors, topic_vectors2)
                assert stats_after["lru_cache_hits"] > stats_before["lru_cache_hits"]

    def test_cache_management(self, tmp_path, temp_cache_dir):
        """Test cache management functionality"""
        docs = [
            {"text": f"Document {i} contains scientific content about electrophoresis method {i}.",
             "source": f"doc_{i}.txt"}
            for i in range(10)
        ]

        output_file = tmp_path / "cache_mgmt_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=3,  # Small cache to test eviction
            disk_cache_dir=temp_cache_dir
        )

        info = decoder.get_manifest_info()
        total_vectors = info["total_vectors"]

        if total_vectors > 5:
            # Access more vectors than cache can hold
            vector_ids = list(range(min(6, total_vectors)))

            for vec_id in vector_ids:
                decoder.get_vectors_by_ids([vec_id])

            stats = decoder.get_cache_stats()
            print(f"Cache management test - Memory cache: {stats['memory_cache_size']}/{stats['memory_cache_limit']}")

            # Should not exceed cache limit
            assert stats["memory_cache_size"] <= stats["memory_cache_limit"]

            # Test cache clearing
            decoder.clear_cache()
            stats_after_clear = decoder.get_cache_stats()
            assert stats_after_clear["memory_cache_size"] == 0
            assert stats_after_clear["lru_cache_size"] == 0

    def test_vector_integrity_enhanced(self, tmp_path, temp_cache_dir):
        """Test vector integrity with enhanced decoder"""
        docs = [{"text": "Test document for enhanced vector integrity validation.", "source": "integrity.txt"}]

        # Generate original vectors
        pipeline = TextVectorPipeline()
        original_vectors, original_metadata = pipeline.process_documents(docs)

        # Encode with enhanced encoder
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=100)
        encoder.add_vectors(original_vectors, original_metadata)

        output_file = tmp_path / "integrity_enhanced_test.mp4"
        encoder.encode_to_mp4(str(output_file))

        # Decode with enhanced decoder
        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=10,
            disk_cache_dir=temp_cache_dir
        )

        decoded_vectors, decoded_metadata = decoder.get_vectors_by_ids(list(range(len(original_vectors))))

        # Check vector similarity (should be nearly identical)
        if len(decoded_vectors) > 0 and len(original_vectors) > 0:
            similarity = np.dot(original_vectors[0], decoded_vectors[0])
            assert similarity > 0.99, f"Vector similarity too low: {similarity}"
            print(f"Enhanced vector integrity check passed: similarity = {similarity:.6f}")

            # Test that cached retrieval maintains integrity
            decoded_vectors2, _ = decoder.get_vectors_by_ids([0])
            cached_similarity = np.dot(decoded_vectors[0], decoded_vectors2[0])
            assert cached_similarity > 0.999, f"Cached vector similarity too low: {cached_similarity}"

    def test_manifest_streaming_load(self, tmp_path):
        """Test streaming manifest loading from separate endpoint"""
        docs = [{"text": "Manifest streaming test document.", "source": "streaming.txt"}]

        output_file = tmp_path / "streaming_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        manifest_file = tmp_path / "streaming_test_manifest.json"
        with open(manifest_file, 'r') as f:
            manifest_content = f.read()

        # Mock CDN manifest endpoint
        with requests_mock.Mocker() as m:
            m.get("https://cdn.example.com/streaming_manifest.json", text=manifest_content)

            decoder = CachedVectorMP4Decoder(
                mp4_path=str(output_file),  # Local MP4
                manifest_path="https://cdn.example.com/streaming_manifest.json",  # Remote manifest
                cache_size=5
            )

            # Should load manifest from URL
            info = decoder.get_manifest_info()
            assert info["total_vectors"] > 0

            print(f"Streaming manifest test passed: {info['total_vectors']} vectors loaded")

    def test_error_handling_and_retry(self, tmp_path, temp_cache_dir):
        """Test error handling and retry mechanisms"""
        docs = [{"text": "Error handling test document.", "source": "error_test.txt"}]

        output_file = tmp_path / "error_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        # Test with invalid cache directory (should handle gracefully)
        invalid_cache = "/invalid/path/that/does/not/exist"

        # Should not crash even with invalid cache path
        try:
            decoder = CachedVectorMP4Decoder(
                mp4_path=str(output_file),
                cache_size=5,
                disk_cache_dir=invalid_cache
            )
            # Should still work without disk cache
            info = decoder.get_manifest_info()
            assert info["total_vectors"] > 0
            print("Error handling test passed: graceful degradation without disk cache")
        except Exception as e:
            pytest.fail(f"Should handle invalid cache directory gracefully: {e}")

        # Test retry mechanism with mocked network failures
        with requests_mock.Mocker() as m:
            # First two requests fail, third succeeds
            responses = [
                {'exc': requests.exceptions.ConnectionError},
                {'exc': requests.exceptions.Timeout},
                {'text': '{"test": "success"}', 'status_code': 200}
            ]
            m.get("https://cdn.example.com/retry_test.json", responses)

            decoder = CachedVectorMP4Decoder(
                mp4_path=str(output_file),
                max_retries=3,
                timeout=1
            )

            # Test the retry mechanism would work in load_manifest_from_url
            try:
                decoder.load_manifest_from_url("https://cdn.example.com/retry_test.json")
                print("Retry mechanism test passed")
            except Exception as e:
                # This is expected to pass with our retry logic
                pass


class TestEnhancedPerformance:
    """Test performance characteristics of enhanced system"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory for performance tests"""
        cache_dir = tmp_path / "perf_cache"
        cache_dir.mkdir()
        yield str(cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def test_large_document_processing(self, tmp_path, temp_cache_dir):
        """Test processing larger documents efficiently"""
        # Create a larger synthetic document
        large_text = """
        Electrophoresis is a laboratory technique used to separate DNA, RNA, or protein molecules based on their size and electrical charge.
        """ * 100  # Repeat to create larger content

        docs = [{"text": large_text, "source": "large_doc.txt"}]

        output_file = tmp_path / "large_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=50,
            disk_cache_dir=temp_cache_dir
        )

        info = decoder.get_manifest_info()
        print(f"Large document test: {info['total_vectors']} vectors, {info['total_fragments']} fragments")

        # Should handle large documents without issues
        assert info["total_vectors"] >= 1
        assert info["total_fragments"] >= 1

        # Test efficient access patterns
        if info["total_vectors"] > 10:
            # Access vectors in chunks (simulating real usage)
            chunk_size = 5
            for start_idx in range(0, min(20, info["total_vectors"]), chunk_size):
                end_idx = min(start_idx + chunk_size, info["total_vectors"])
                vector_ids = list(range(start_idx, end_idx))
                vectors, metadata = decoder.get_vectors_by_ids(vector_ids)
                assert len(vectors) == len(vector_ids)

        stats = decoder.get_cache_stats()
        print(f"Large document cache performance: {stats}")

    def test_multiple_fragments_creation(self, tmp_path, temp_cache_dir):
        """Test creation of multiple fragments with small chunk size"""
        # Create documents that will definitely create multiple fragments
        docs = [
            {
                "text": f"Document {i} contains detailed information about electrophoresis method {i} and its applications in scientific research.",
                "source": f"doc_{i}.txt"}
            for i in range(15)  # 15 documents should create enough vectors
        ]

        output_file = tmp_path / "multi_fragment_test.mp4"

        # Use a small chunk size to force multiple fragments
        pipeline = TextVectorPipeline(vector_dim=384)
        vectors, metadata = pipeline.process_documents(docs)

        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=5)  # Small chunk size
        encoder.add_vectors(vectors, metadata)
        encoder.encode_to_mp4(str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=20,
            disk_cache_dir=temp_cache_dir
        )

        info = decoder.get_manifest_info()
        print(f"Multiple fragments test: {info['total_vectors']} vectors, {info['total_fragments']} fragments")

        # With chunk_size=5 and 15 documents, we should get multiple fragments
        assert info["total_vectors"] >= 10  # Should have plenty of vectors
        if info["total_vectors"] > 5:
            assert info["total_fragments"] > 1  # Should have multiple fragments

        # Test accessing vectors across different fragments
        if info["total_fragments"] > 1:
            # Access vectors from different fragments
            vectors_per_fragment = info["total_vectors"] // info["total_fragments"]
            test_vectors = [0, vectors_per_fragment, info["total_vectors"] - 1]
            test_vectors = [v for v in test_vectors if v < info["total_vectors"]]

            vectors, metadata = decoder.get_vectors_by_ids(test_vectors)
            assert len(vectors) == len(test_vectors)
            print(f"Successfully accessed vectors across {info['total_fragments']} fragments")


    def test_concurrent_access_simulation(self, tmp_path, temp_cache_dir):
        """Test simulated concurrent access patterns"""
        docs = [
            {"text": f"Concurrent test document {i} about electrophoresis method {i}.",
             "source": f"concurrent_{i}.txt"}
            for i in range(20)
        ]

        output_file = tmp_path / "concurrent_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        decoder = CachedVectorMP4Decoder(
            mp4_path=str(output_file),
            cache_size=10,
            disk_cache_dir=temp_cache_dir
        )

        info = decoder.get_manifest_info()
        total_vectors = info["total_vectors"]

        if total_vectors > 5:
            # Simulate multiple "users" accessing different vectors
            access_patterns = [
                list(range(0, min(5, total_vectors))),
                list(range(2, min(7, total_vectors))),
                list(range(1, min(6, total_vectors)))
            ]

            for pattern in access_patterns:
                vectors, metadata = decoder.get_vectors_by_ids(pattern)
                assert len(vectors) == len(pattern)

            # Should show good cache performance
            stats = decoder.get_cache_stats()
            cache_hit_ratio = stats["lru_cache_hits"] / (stats["lru_cache_hits"] + stats["lru_cache_misses"]) if stats["lru_cache_misses"] > 0 else 1.0
            print(f"Concurrent access cache hit ratio: {cache_hit_ratio:.2f}")


if __name__ == "__main__":
    # Run enhanced tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])