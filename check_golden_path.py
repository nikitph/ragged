#!/usr/bin/env python3
"""
Standalone Golden Path Test for Ragged Video System
Run this to get a detailed report of your system's status

Usage: python check_golden_path.py
"""

import tempfile
import time
import json
from pathlib import Path
import sys
import traceback


def print_banner():
    """Print a nice banner"""
    print("🎯 Ragged Video System - Golden Path Analysis")
    print("=" * 60)
    print("Testing: Text → Video → Search → Results")
    print("=" * 60)


def test_imports():
    """Test all required imports"""
    print("\n📦 TESTING IMPORTS")
    print("-" * 30)

    imports = [
        ("ragged.video", "VideoEncoder, VideoRetriever"),
        ("ragged.video.config", "get_default_config"),
        ("ragged.video.utils", "encode_to_qr, decode_qr, qr_to_frame"),
        ("cv2", "OpenCV"),
        ("faiss", "FAISS"),
        ("sentence_transformers", "SentenceTransformer"),
        ("qrcode", "QR Code library"),
        ("PIL", "Pillow"),
    ]

    failed_imports = []

    for module, description in imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"\n⚠️  Failed imports: {failed_imports}")
        return False

    print("✅ All imports successful")
    return True


def test_configuration():
    """Test configuration system"""
    print("\n🔧 TESTING CONFIGURATION")
    print("-" * 30)

    try:
        from ragged.video.config import get_default_config

        config = get_default_config()

        required_sections = ['embedding', 'qr', 'chunking', 'index', 'retrieval']
        missing_sections = [section for section in required_sections if section not in config]

        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False

        print("✅ Configuration structure valid")

        # Test specific config values
        print(f"   - Embedding model: {config['embedding']['model']}")
        print(f"   - Chunk size: {config['chunking']['chunk_size']}")
        print(f"   - QR version: {config['qr']['version']}")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_qr_processing():
    """Test QR code round-trip"""
    print("\n📱 TESTING QR PROCESSING")
    print("-" * 30)

    try:
        from ragged.video.utils import encode_to_qr, decode_qr, qr_to_frame

        test_cases = [
            "Short text",
            "Medium length text with some special characters: àéîöù",
            "Longer text that should test compression: " + "A" * 200,
            '{"json": "data", "number": 42, "array": [1,2,3]}'
        ]

        success_count = 0

        for i, test_data in enumerate(test_cases):
            try:
                # Encode
                qr_image = encode_to_qr(test_data)
                if qr_image is None:
                    print(f"❌ Test {i + 1}: QR encoding failed")
                    continue

                # Convert to frame
                frame = qr_to_frame(qr_image, (256, 256))
                if frame is None:
                    print(f"❌ Test {i + 1}: Frame conversion failed")
                    continue

                # Decode
                decoded = decode_qr(frame)
                if decoded == test_data:
                    print(f"✅ Test {i + 1}: Round-trip successful ({len(test_data)} chars)")
                    success_count += 1
                else:
                    print(f"❌ Test {i + 1}: Decode mismatch")
                    print(f"   Expected: {test_data[:50]}...")
                    print(f"   Got: {decoded[:50] if decoded else None}...")

            except Exception as e:
                print(f"❌ Test {i + 1}: Exception - {e}")

        success_rate = success_count / len(test_cases)
        print(f"\n📊 QR Success Rate: {success_count}/{len(test_cases)} ({success_rate * 100:.1f}%)")

        return success_rate >= 0.75  # 75% success rate acceptable

    except Exception as e:
        print(f"❌ QR processing error: {e}")
        return False


def test_video_encoding():
    """Test video encoding functionality"""
    print("\n🎥 TESTING VIDEO ENCODING")
    print("-" * 30)

    try:
        from ragged.video import VideoEncoder

        # Test chunks
        chunks = [
            "Machine learning algorithms process data automatically.",
            "Artificial intelligence enables smart automation systems.",
            "Deep learning uses neural networks for pattern recognition.",
            "Natural language processing helps computers understand text.",
            "Computer vision analyzes images and video content."
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create encoder
            encoder = VideoEncoder()
            encoder.add_chunks(chunks)

            # Check stats
            stats = encoder.get_stats()
            print(f"✅ Encoder created with {stats['total_chunks']} chunks")

            # Build video
            video_file = temp_path / "test.mp4"
            index_file = temp_path / "test_index.json"

            start_time = time.time()
            build_stats = encoder.build_video(
                str(video_file),
                str(index_file),
                codec="mp4v",
                show_progress=False
            )
            encoding_time = time.time() - start_time

            # Check outputs
            if not video_file.exists():
                print("❌ Video file not created")
                return False

            if not index_file.exists():
                print("❌ Index file not created")
                return False

            faiss_file = temp_path / "test_index.faiss"
            if not faiss_file.exists():
                print("❌ FAISS index file not created")
                return False

            # Report stats
            print(f"✅ Video created successfully")
            print(f"   - Encoding time: {encoding_time:.2f}s")
            print(f"   - Video size: {build_stats.get('video_size_mb', 0):.2f} MB")
            print(f"   - Total frames: {build_stats.get('total_frames', 0)}")
            print(f"   - Backend: {build_stats.get('backend', 'unknown')}")

            return True

    except Exception as e:
        print(f"❌ Video encoding error: {e}")
        traceback.print_exc()
        return False


def test_video_retrieval():
    """Test video retrieval and search"""
    print("\n🔍 TESTING VIDEO RETRIEVAL")
    print("-" * 30)

    try:
        from ragged.video import VideoEncoder, VideoRetriever

        # Create test video
        chunks = [
            "Python programming language is known for simplicity and readability.",
            "JavaScript powers modern web applications and user interfaces.",
            "Machine learning algorithms analyze patterns in large datasets.",
            "Cloud computing provides scalable infrastructure for applications.",
            "Database optimization improves application performance significantly."
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Encode video
            encoder = VideoEncoder()
            encoder.add_chunks(chunks)

            video_file = temp_path / "retrieval_test.mp4"
            index_file = temp_path / "retrieval_test_index.json"

            encoder.build_video(str(video_file), str(index_file), codec="mp4v", show_progress=False)

            # Test retrieval
            retriever = VideoRetriever(str(video_file), str(index_file))

            # Test searches
            search_tests = [
                ("Python", ["Python", "programming"]),
                ("web development", ["JavaScript", "web"]),
                ("data analysis", ["machine learning", "datasets"]),
                ("infrastructure", ["cloud", "scalable"]),
                ("performance", ["database", "optimization"])
            ]

            successful_searches = 0

            for query, expected_keywords in search_tests:
                start_time = time.time()
                results = retriever.search(query, top_k=2)
                search_time = time.time() - start_time

                if len(results) > 0:
                    # Check if results contain expected keywords
                    found_relevant = False
                    for result in results:
                        if any(keyword.lower() in result.lower() for keyword in expected_keywords):
                            found_relevant = True
                            break

                    if found_relevant:
                        print(f"✅ Query '{query}': found relevant results ({search_time:.3f}s)")
                        successful_searches += 1
                    else:
                        print(f"⚠️  Query '{query}': no relevant results")
                        print(f"   Results: {[r[:40] + '...' for r in results]}")
                else:
                    print(f"❌ Query '{query}': no results found")

            print(
                f"\n📊 Search Success Rate: {successful_searches}/{len(search_tests)} ({successful_searches / len(search_tests) * 100:.1f}%)")

            # Test chunk retrieval
            try:
                chunk_0 = retriever.get_chunk_by_id(0)
                if chunk_0:
                    print(f"✅ Chunk retrieval working")
                else:
                    print(f"⚠️  Chunk retrieval returned None")
            except Exception as e:
                print(f"❌ Chunk retrieval failed: {e}")

            return successful_searches >= len(search_tests) * 0.6  # 60% success rate

    except Exception as e:
        print(f"❌ Video retrieval error: {e}")
        traceback.print_exc()
        return False


def test_integrity():
    """Test video integrity validation"""
    print("\n🔍 TESTING VIDEO INTEGRITY")
    print("-" * 30)

    try:
        from ragged.video import VideoEncoder, VideoRetriever

        chunks = [f"Integrity test chunk {i} with content" for i in range(10)]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create video
            encoder = VideoEncoder()
            encoder.add_chunks(chunks)

            video_file = temp_path / "integrity_test.mp4"
            index_file = temp_path / "integrity_test_index.json"

            encoder.build_video(str(video_file), str(index_file), codec="mp4v", show_progress=False)

            # Test integrity
            retriever = VideoRetriever(str(video_file), str(index_file))
            integrity = retriever.validate_integrity()

            print(f"✅ Integrity check completed")
            print(f"   - Integrity: {integrity['integrity_percent']:.1f}%")
            print(f"   - Healthy: {integrity['is_healthy']}")
            print(f"   - Issues: {len(integrity['issues'])}")

            if integrity['issues']:
                print("   - Sample issues:")
                for issue in integrity['issues'][:3]:
                    print(f"     • {issue}")

            return integrity['integrity_percent'] >= 80  # 80% integrity acceptable

    except Exception as e:
        print(f"❌ Integrity test error: {e}")
        return False


def generate_report(results):
    """Generate final report"""
    print("\n" + "=" * 60)
    print("📊 GOLDEN PATH ANALYSIS REPORT")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    success_rate = passed_tests / total_tests * 100

    print(f"\nOverall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

    print(f"\nDetailed Results:")
    status_icons = {True: "✅", False: "❌"}
    for test_name, passed in results.items():
        print(f"  {status_icons[passed]} {test_name}")

    print(f"\n" + "-" * 60)

    if success_rate >= 80:
        print("🎉 GOLDEN PATH STATUS: WORKING WELL")
        print("✅ Your system is ready for production use!")
        print("\nRecommendations:")
        print("- Continue with regular testing")
        print("- Monitor performance in production")
        print("- Consider adding more edge case tests")

    elif success_rate >= 60:
        print("⚠️  GOLDEN PATH STATUS: MOSTLY WORKING")
        print("🔧 Your system works but has some issues")
        print("\nRecommendations:")
        print("- Fix failing components before production")
        print("- Focus on QR processing if that's failing")
        print("- Add error handling for edge cases")

    else:
        print("🚧 GOLDEN PATH STATUS: NEEDS WORK")
        print("❌ Multiple components are failing")
        print("\nRecommendations:")
        print("- Fix critical issues before proceeding")
        print("- Check dependencies and configuration")
        print("- Review implementation for basic functionality")

    print(f"\n" + "=" * 60)


def main():
    """Main test runner"""
    print_banner()

    # Run all tests
    results = {}

    results["Imports"] = test_imports()
    if not results["Imports"]:
        print("\n❌ Cannot proceed without proper imports")
        return 1

    results["Configuration"] = test_configuration()
    results["QR Processing"] = test_qr_processing()
    results["Video Encoding"] = test_video_encoding()
    results["Video Retrieval"] = test_video_retrieval()
    results["Integrity Check"] = test_integrity()

    # Generate final report
    generate_report(results)

    # Return appropriate exit code
    success_rate = sum(1 for result in results.values() if result) / len(results)
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)