#!/usr/bin/env python3
"""
Test script for Vector MP4 Encoder/Decoder
"""

import numpy as np
import os
import time
import json
import struct
from typing import List, Dict


def create_test_data(num_vectors: int = 10000, vector_dim: int = 1536) -> tuple:
    """Create realistic test data"""
    print(f"Creating {num_vectors} test vectors of dimension {vector_dim}")

    # Create somewhat realistic embeddings (not completely random)
    np.random.seed(42)
    base_vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)

    # Normalize vectors (common for embeddings)
    norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    vectors = base_vectors / norms

    # Create metadata with realistic domains and topics
    domains = ["legal", "medical", "technical", "finance", "research"]
    topics = {
        "legal": ["contracts", "liability", "intellectual_property", "compliance", "litigation"],
        "medical": ["diagnosis", "treatment", "symptoms", "medication", "surgery"],
        "technical": ["api", "database", "security", "performance", "architecture"],
        "finance": ["investment", "risk", "portfolio", "trading", "analysis"],
        "research": ["methodology", "results", "hypothesis", "literature", "conclusion"]
    }

    metadata = []
    for i in range(num_vectors):
        domain = domains[i % len(domains)]
        topic = topics[domain][i % len(topics[domain])]
        metadata.append({
            "doc_id": f"doc_{i:06d}",
            "domain": domain,
            "topic": topic,
            "title": f"Document about {topic} #{i}",
            "content_preview": f"This is document {i} discussing {topic} in the {domain} domain...",
            "timestamp": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "author": f"author_{i % 10}",
            "source": f"source_{domain}_{i % 5}"
        })

    return vectors, metadata


def test_basic_encoding_decoding():
    """Test basic encode/decode functionality"""
    print("\n=== Testing Basic Encoding/Decoding ===")

    # Create test data
    vectors, metadata = create_test_data(100000, 1536)

    # Test encoding
    print("Encoding vectors...")
    encoder = VectorMP4Encoder(vector_dim=1536, chunk_size=50)
    encoder.add_vectors(vectors, metadata)

    # Encode to MP4
    test_file = "test_knowledge_base.mp4"
    start_time = time.time()
    encoder.encode_to_mp4(test_file)
    encoding_time = time.time() - start_time

    print(f"Encoding completed in {encoding_time:.2f} seconds")
    print(f"File size: {os.path.getsize(test_file) / 1024 / 1024:.2f} MB")

    # Test decoding
    print("\nDecoding vectors...")
    start_time = time.time()
    decoder = VectorMP4Decoder(test_file)
    decoding_time = time.time() - start_time

    print(f"Decoder initialized in {decoding_time:.2f} seconds")
    print(f"Total vectors in manifest: {decoder.manifest['metadata']['total_vectors']}")
    print(f"Number of fragments: {len(decoder.manifest['metadata']['fragments'])}")

    return vectors, metadata, decoder


def test_vector_retrieval(original_vectors, original_metadata, decoder):
    """Test vector retrieval functionality"""
    print("\n=== Testing Vector Retrieval ===")

    # Test 1: Get vectors by IDs
    test_ids = [0, 5, 10, 15, 20]
    print(f"Retrieving vectors by IDs: {test_ids}")

    retrieved_vectors, retrieved_metadata = decoder.get_vectors_by_ids(test_ids)
    print(f"Retrieved {len(retrieved_vectors)} vectors")

    # Verify integrity
    print("Checking vector integrity:")
    for i, vec_id in enumerate(test_ids):
        if vec_id < len(original_vectors):
            original_vec = original_vectors[vec_id]
            retrieved_vec = retrieved_vectors[i]
            similarity = np.dot(original_vec, retrieved_vec)
            print(f"  Vector {vec_id}: similarity = {similarity:.6f} (should be ~1.0)")

            # Check metadata
            orig_topic = original_metadata[vec_id]['topic']
            retr_topic = retrieved_metadata[i]['topic']
            print(f"  Metadata match: {orig_topic} == {retr_topic} -> {orig_topic == retr_topic}")

            if similarity < 0.99:
                print(f"  WARNING: Low similarity for vector {vec_id}")
            if orig_topic != retr_topic:
                print(f"  WARNING: Metadata mismatch for vector {vec_id}")

    # Test 2: Get vectors by topic
    print(f"\nTesting topic-based retrieval...")
    topic_vectors, topic_metadata = decoder.get_vectors_by_topic("contracts")
    print(f"Retrieved {len(topic_vectors)} vectors with topic 'contracts'")

    if len(topic_metadata) > 0:
        print(f"Sample topics: {[m['topic'] for m in topic_metadata[:5]]}")


def test_search_functionality(decoder):
    """Test search functionality"""
    print("\n=== Testing Search Functionality ===")

    # Create a random query vector
    np.random.seed(123)
    query_vector = np.random.randn(1536).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize

    # Test search across all topics
    print("Searching across all topics...")
    start_time = time.time()
    results = decoder.search_vectors(query_vector, top_k=5)
    search_time = time.time() - start_time

    print(f"Search completed in {search_time:.3f} seconds")
    print(f"Top {len(results)} results:")
    for i, result in enumerate(results):
        meta = result['metadata']
        sim = result['similarity']
        print(f"  {i + 1}. [{meta['domain']}] {meta['topic']} - {meta['title']} (sim: {sim:.3f})")

    # Test topic-specific search
    print(f"\nSearching within 'contracts' topic...")
    contract_results = decoder.search_vectors(query_vector, top_k=3, topic="contracts")
    print(f"Contract topic results:")
    for i, result in enumerate(contract_results):
        meta = result['metadata']
        sim = result['similarity']
        print(f"  {i + 1}. {meta['topic']} - {meta['title']} (sim: {sim:.3f})")


def test_manifest_structure(decoder):
    """Test manifest structure and integrity"""
    print("\n=== Testing Manifest Structure ===")

    manifest = decoder.manifest

    print(f"Manifest metadata:")
    print(f"  Vector dimension: {manifest['metadata']['vector_dim']}")
    print(f"  Chunk size: {manifest['metadata']['chunk_size']}")
    print(f"  Total vectors: {manifest['metadata']['total_vectors']}")
    print(f"  Number of fragments: {len(manifest['metadata']['fragments'])}")

    print(f"\nFragment details:")
    for frag in manifest['metadata']['fragments']:
        print(f"  Fragment {frag['id']}: vectors={frag['vector_count']}, topics={frag['topics'][:3]}")

    print(f"\nVector mapping sample:")
    vector_map = manifest['vector_map']
    sample_keys = list(vector_map.keys())[:5]
    for key in sample_keys:
        vec_info = vector_map[key]
        print(f"  Vector {key}: fragment={vec_info['fragment_id']}, "
              f"offset={vec_info['local_offset']}, topic={vec_info['metadata']['topic']}")


def test_file_structure(filename):
    """Test MP4 file structure"""
    print("\n=== Testing MP4 File Structure ===")

    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return

    # Read and analyze MP4 boxes
    with open(filename, 'rb') as f:
        boxes_found = []
        total_size = 0

        while True:
            header = f.read(8)
            if len(header) < 8:
                break

            box_size, box_type = struct.unpack('>I4s', header)
            box_type = box_type.decode('ascii', errors='ignore')
            boxes_found.append((box_type, box_size))
            total_size += box_size

            # Skip box content
            f.seek(box_size - 8, 1)

    print(f"MP4 boxes found:")
    for box_type, size in boxes_found:
        print(f"  {box_type}: {size} bytes ({size / 1024:.1f} KB)")

    print(f"Total file size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")


def test_performance_benchmark():
    """Test performance with different sizes"""
    print("\n=== Performance Benchmark ===")

    sizes = [50, 100, 200]
    results = []

    for size in sizes:
        print(f"\nTesting with {size} vectors...")

        # Create data
        vectors, metadata = create_test_data(size, 1536)

        # Benchmark encoding
        encoder = VectorMP4Encoder(vector_dim=1536, chunk_size=min(50, size // 2))
        encoder.add_vectors(vectors, metadata)

        filename = f"benchmark_{size}.mp4"
        start_time = time.time()
        encoder.encode_to_mp4(filename)
        encoding_time = time.time() - start_time

        file_size = os.path.getsize(filename)

        # Benchmark decoding
        start_time = time.time()
        decoder = VectorMP4Decoder(filename)
        decoding_time = time.time() - start_time

        # Benchmark search
        query_vector = np.random.randn(1536).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

        start_time = time.time()
        search_results = decoder.search_vectors(query_vector, top_k=5)
        search_time = time.time() - start_time

        results.append({
            "vectors": size,
            "file_size_mb": file_size / 1024 / 1024,
            "encoding_time": encoding_time,
            "decoding_time": decoding_time,
            "search_time": search_time,
            "vectors_per_mb": size / (file_size / 1024 / 1024)
        })

        print(f"  Encoded in {encoding_time:.3f}s, size: {file_size / 1024:.1f}KB")
        print(f"  Decoded in {decoding_time:.3f}s")
        print(f"  Search in {search_time:.3f}s")

        # Cleanup
        os.remove(filename)
        manifest_file = filename.replace('.mp4', '_manifest.json')
        if os.path.exists(manifest_file):
            os.remove(manifest_file)
        faiss_file = filename.replace('.mp4', '_faiss.index')
        if os.path.exists(faiss_file):
            os.remove(faiss_file)

    print(f"\nPerformance Summary:")
    print(f"{'Vectors':<8} {'Size(MB)':<10} {'Encode(s)':<10} {'Decode(s)':<10} {'Search(s)':<10} {'Vec/MB':<8}")
    print("-" * 70)
    for result in results:
        print(f"{result['vectors']:<8} {result['file_size_mb']:<10.2f} "
              f"{result['encoding_time']:<10.3f} {result['decoding_time']:<10.3f} "
              f"{result['search_time']:<10.3f} {result['vectors_per_mb']:<8.0f}")


def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        "test_knowledge_base.mp4",
        "test_knowledge_base_manifest.json"
    ]

    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up {filename}")


def main():
    """Run all tests"""
    print("Starting Vector MP4 Codec Tests")
    print("=" * 50)

    try:
        # Run tests
        vectors, metadata, decoder = test_basic_encoding_decoding()
        test_vector_retrieval(vectors, metadata, decoder)
        test_search_functionality(decoder)
        test_manifest_structure(decoder)
        test_file_structure("test_knowledge_base.mp4")
        test_performance_benchmark()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

   # finally:
   #     cleanup_test_files()


if __name__ == "__main__":
    # Check if we need to create the encoder/decoder classes for testing
    try:
        from encoder import VectorMP4Encoder
        from decoder import  VectorMP4Decoder
    except ImportError:
        print("Error: vector_mp4_codec.py not found. Please ensure both files are in the same directory.")
        print("You can copy the encoder/decoder code from the first artifact into vector_mp4_codec.py")
        exit(1)

    main()