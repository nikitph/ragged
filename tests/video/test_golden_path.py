#!/usr/bin/env python3
"""
Golden Path Verification Test for Ragged Video System
Tests the complete workflow: Text → Video → Search → Results

Run this to check if your golden path is working end-to-end
"""

import tempfile
import time
from pathlib import Path
import sys
import traceback


def test_golden_path():
    """Test the complete golden path workflow"""

    print("🚀 Testing Ragged Video Golden Path")
    print("=" * 50)

    try:
        # Import required modules
        print("📦 Importing modules...")
        from ragged.video import VideoEncoder, VideoRetriever
        from ragged.video.config import get_default_config
        print("✅ Imports successful")

        # Test configuration
        print("\n🔧 Testing configuration...")
        config = get_default_config()
        required_keys = ['embedding', 'qr', 'chunking', 'index']
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            print(f"❌ Missing config keys: {missing_keys}")
            return False
        print("✅ Configuration valid")

        # Test QR functionality
        print("\n📱 Testing QR processing...")
        from ragged.video.utils import encode_to_qr, decode_qr, qr_to_frame

        test_qr_data = "Golden path test data"
        qr_image = encode_to_qr(test_qr_data)

        if qr_image is None:
            print("❌ QR encoding failed")
            return False

        frame = qr_to_frame(qr_image, (256, 256))
        decoded = decode_qr(frame)

        if decoded != test_qr_data:
            print(f"❌ QR round-trip failed: got '{decoded}', expected '{test_qr_data}'")
            print("   This is the main issue blocking your golden path!")
            return False
        print("✅ QR processing working")

        # Test video encoding
        print("\n🎥 Testing video encoding...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Sample knowledge base
            chunks = [
                "Machine learning is a powerful tool for data analysis.",
                "Artificial intelligence can automate complex tasks.",
                "Deep learning uses neural networks for pattern recognition."
            ]

            # Create encoder
            encoder = VideoEncoder()
            encoder.add_chunks(chunks)

            # Build video
            video_file = temp_path / "golden_test.mp4"
            index_file = temp_path / "golden_test_index.json"

            start_time = time.time()
            build_stats = encoder.build_video(
                str(video_file),
                str(index_file),
                codec="mp4v",
                show_progress=False
            )
            encoding_time = time.time() - start_time

            print(f"✅ Video encoded in {encoding_time:.2f}s")
            print(f"   - Chunks: {build_stats['total_chunks']}")
            print(f"   - Size: {build_stats.get('video_size_mb', 0):.2f} MB")

            # Verify files exist
            if not video_file.exists():
                print("❌ Video file not created")
                return False

            if not index_file.exists():
                print("❌ Index file not created")
                return False

            # Test video retrieval
            print("\n🔍 Testing video retrieval...")
            retriever = VideoRetriever(str(video_file), str(index_file))

            # Test search
            search_query = "machine learning"
            start_time = time.time()
            results = retriever.search(search_query, top_k=2)
            search_time = time.time() - start_time

            print(f"✅ Search completed in {search_time:.3f}s")
            print(f"   - Query: '{search_query}'")
            print(f"   - Results: {len(results)}")

            if len(results) == 0:
                print("⚠️  No search results found - search may not be working properly")
                return False

            # Display results
            for i, result in enumerate(results):
                print(f"   - Result {i + 1}: {result[:60]}...")

            # Test chunk retrieval
            print("\n📄 Testing chunk retrieval...")
            chunk_0 = retriever.get_chunk_by_id(0)

            if chunk_0 is None:
                print("❌ Chunk retrieval failed")
                return False

            print(f"✅ Retrieved chunk 0: {chunk_0[:50]}...")

            # Test integrity
            print("\n🔍 Testing video integrity...")
            try:
                integrity = retriever.validate_integrity()

                if integrity['is_healthy']:
                    print(f"✅ Video integrity: {integrity['integrity_percent']:.1f}%")
                else:
                    print(f"⚠️  Video integrity issues: {integrity['integrity_percent']:.1f}%")
                    print(f"   Issues found: {len(integrity['issues'])}")
                    for issue in integrity['issues'][:3]:  # Show first 3 issues
                        print(f"   - {issue}")

                    if integrity['integrity_percent'] < 50:
                        print("❌ Critical integrity issues")
                        return False

            except Exception as e:
                print(f"⚠️  Integrity check failed: {e}")
                # Don't fail the golden path for integrity issues

        print("\n" + "=" * 50)
        print("🎉 GOLDEN PATH WORKING!")
        print("✅ All core functionality operational")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Checking dependencies...")

    deps = [
        ("opencv-python", "cv2"),
        ("faiss-cpu", "faiss"),
        ("sentence-transformers", "sentence_transformers"),
        ("qrcode", "qrcode"),
        ("pillow", "PIL"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm")
    ]

    missing = []
    for dep_name, import_name in deps:
        try:
            __import__(import_name)
            print(f"✅ {dep_name}")
        except ImportError:
            print(f"❌ {dep_name}")
            missing.append(dep_name)

    if missing:
        print(f"\n⚠️  Missing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False

    print("✅ All dependencies available")
    return True


def main():
    """Main test runner"""
    print("Ragged Video System - Golden Path Test")
    print("=" * 60)

    # Check dependencies first
    if not test_dependencies():
        return 1

    print()

    # Test golden path
    if test_golden_path():
        print("\n🚀 Your golden path is working!")
        print("You can proceed with confidence.")
        return 0
    else:
        print("\n🚧 Golden path has issues that need fixing.")
        print("Focus on the failed components above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)