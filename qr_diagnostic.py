#!/usr/bin/env python3
"""
QR Code Diagnostic Tool for Ragged Video System
Diagnoses and fixes QR code processing issues

Usage: python qr_diagnostic.py
"""

import tempfile
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys


def test_qr_libraries():
    """Test different QR libraries and configurations"""
    print("🔍 TESTING QR LIBRARIES & CONFIGURATIONS")
    print("=" * 50)

    test_data = "Hello QR World"

    # Test 1: qrcode library with different settings
    print("\n📱 Testing qrcode library...")
    try:
        import qrcode

        # Try different configurations
        configs = [
            {"version": 1, "error_correction": qrcode.constants.ERROR_CORRECT_L, "box_size": 10, "border": 4},
            {"version": 1, "error_correction": qrcode.constants.ERROR_CORRECT_M, "box_size": 10, "border": 4},
            {"version": 1, "error_correction": qrcode.constants.ERROR_CORRECT_H, "box_size": 10, "border": 4},
            {"version": 5, "error_correction": qrcode.constants.ERROR_CORRECT_M, "box_size": 5, "border": 2},
        ]

        for i, config in enumerate(configs):
            try:
                qr = qrcode.QRCode(**config)
                qr.add_data(test_data)
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")
                print(f"✅ Config {i + 1}: Generated QR code")

                # Test decoding this QR
                test_decode_qr_image(img, test_data, f"Config {i + 1}")

            except Exception as e:
                print(f"❌ Config {i + 1}: {e}")

    except ImportError:
        print("❌ qrcode library not available")


def test_decode_qr_image(qr_image, expected_data, config_name):
    """Test decoding a QR image with different methods"""

    # Convert PIL image to numpy array for OpenCV
    if hasattr(qr_image, 'convert'):
        # PIL Image
        img_array = np.array(qr_image.convert('RGB'))
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = qr_image

    # Method 1: OpenCV QR decoder
    try:
        detector = cv2.QRCodeDetector()
        data, vertices_array, binary_qrcode = detector.detectAndDecode(img_bgr)

        if data:
            if data == expected_data:
                print(f"  ✅ {config_name} - OpenCV decode: SUCCESS")
                return True
            else:
                print(f"  ⚠️  {config_name} - OpenCV decode: MISMATCH")
                print(f"     Expected: {expected_data}")
                print(f"     Got: {data}")
        else:
            print(f"  ❌ {config_name} - OpenCV decode: FAILED")

    except Exception as e:
        print(f"  ❌ {config_name} - OpenCV decode error: {e}")

    # Method 2: pyzbar (if available)
    try:
        from pyzbar import pyzbar

        # Convert to grayscale for pyzbar
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray)

        if decoded_objects:
            data = decoded_objects[0].data.decode('utf-8')
            if data == expected_data:
                print(f"  ✅ {config_name} - pyzbar decode: SUCCESS")
                return True
            else:
                print(f"  ⚠️  {config_name} - pyzbar decode: MISMATCH")
        else:
            print(f"  ❌ {config_name} - pyzbar decode: FAILED")

    except ImportError:
        print(f"  ℹ️  {config_name} - pyzbar not available")
    except Exception as e:
        print(f"  ❌ {config_name} - pyzbar error: {e}")

    return False


def analyze_current_implementation():
    """Analyze the current QR implementation"""
    print("\n🔬 ANALYZING CURRENT IMPLEMENTATION")
    print("=" * 50)

    try:
        from ragged.video.utils import encode_to_qr, decode_qr, qr_to_frame

        # Test the current implementation step by step
        test_data = "Debug test data"

        print(f"Testing with data: '{test_data}'")

        # Step 1: encode_to_qr
        print("\n1. Testing encode_to_qr...")
        qr_image = encode_to_qr(test_data)

        if qr_image is None:
            print("❌ encode_to_qr returned None")
            return False
        else:
            print(f"✅ encode_to_qr returned image: {type(qr_image)}")
            if hasattr(qr_image, 'size'):
                print(f"   Image size: {qr_image.size}")

        # Step 2: qr_to_frame
        print("\n2. Testing qr_to_frame...")
        frame_sizes = [(256, 256), (512, 512), (128, 128)]

        for size in frame_sizes:
            frame = qr_to_frame(qr_image, size)

            if frame is None:
                print(f"❌ qr_to_frame({size}) returned None")
                continue
            else:
                print(f"✅ qr_to_frame({size}) returned frame: {frame.shape}")

                # Step 3: decode_qr
                print(f"3. Testing decode_qr with {size} frame...")
                decoded = decode_qr(frame)

                if decoded == test_data:
                    print(f"✅ decode_qr with {size}: SUCCESS!")
                    return True
                elif decoded is None:
                    print(f"❌ decode_qr with {size}: returned None")
                else:
                    print(f"⚠️  decode_qr with {size}: MISMATCH")
                    print(f"   Expected: {test_data}")
                    print(f"   Got: {decoded}")

                # Save frame for manual inspection
                debug_path = f"debug_qr_frame_{size[0]}x{size[1]}.png"
                cv2.imwrite(debug_path, frame)
                print(f"   💾 Saved debug frame: {debug_path}")

        return False

    except Exception as e:
        print(f"❌ Error analyzing implementation: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_video_with_debug():
    """Create a test video and examine the frames"""
    print("\n🎥 CREATING TEST VIDEO WITH DEBUG")
    print("=" * 50)

    try:
        from ragged.video import VideoEncoder

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create simple test video
            chunks = ["Test chunk 1", "Test chunk 2"]

            encoder = VideoEncoder()
            encoder.add_chunks(chunks)

            video_file = temp_path / "debug_test.mp4"
            index_file = temp_path / "debug_test_index.json"

            print("Creating test video...")
            encoder.build_video(str(video_file), str(index_file), codec="mp4v", show_progress=False)

            if not video_file.exists():
                print("❌ Video file not created")
                return False

            print(f"✅ Video created: {video_file}")

            # Extract and examine frames
            print("\nExtracting frames for analysis...")
            cap = cv2.VideoCapture(str(video_file))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                print(f"Frame {frame_count}: shape {frame.shape}")

                # Save frame for inspection
                debug_frame_path = f"debug_video_frame_{frame_count}.png"
                cv2.imwrite(debug_frame_path, frame)
                print(f"  💾 Saved: {debug_frame_path}")

                # Try to decode this frame
                from ragged.video.utils import decode_qr
                decoded = decode_qr(frame)

                if decoded:
                    print(f"  ✅ Decoded: {decoded[:50]}...")
                else:
                    print(f"  ❌ Could not decode frame")

                frame_count += 1

                if frame_count >= 5:  # Limit to first 5 frames
                    break

            cap.release()
            print(f"\nProcessed {frame_count} frames")

            return frame_count > 0

    except Exception as e:
        print(f"❌ Error creating test video: {e}")
        import traceback
        traceback.print_exc()
        return False


def suggest_fixes():
    """Suggest potential fixes based on analysis"""
    print("\n🔧 SUGGESTED FIXES")
    print("=" * 50)

    print("""
Based on the diagnostic results, here are potential fixes:

1. 📱 QR Code Generation Issues:
   - Try different QR versions (1, 5, 10)
   - Adjust error correction levels
   - Modify box size and border settings

2. 🖼️  Frame Conversion Issues:
   - Check image color space (RGB vs BGR)
   - Verify frame dimensions
   - Ensure proper data types (uint8)

3. 🔍 Decoding Issues:
   - Install pyzbar: pip install pyzbar
   - Try different OpenCV QR detector settings
   - Check frame quality and resolution

4. 🎥 Video Processing Issues:
   - Verify video codec compatibility
   - Check frame extraction process
   - Ensure proper video file structure

5. 🔄 Alternative Approach:
   - Consider using text overlays instead of QR codes
   - Implement hybrid approach (QR + metadata)
   - Use different encoding schemes

Next Steps:
1. Run: pip install pyzbar
2. Check the saved debug frames manually
3. Implement the suggested QR settings
4. Test with smaller, simpler data first
""")


def main():
    """Main diagnostic function"""
    print("🔍 QR Code Diagnostic Tool for Ragged Video System")
    print("=" * 60)

    # Test QR libraries and configurations
    test_qr_libraries()

    # Analyze current implementation
    current_works = analyze_current_implementation()

    # Create test video with debug
    video_works = create_test_video_with_debug()

    # Provide suggestions
    suggest_fixes()

    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if current_works:
        print("✅ QR processing is working with some configurations")
    else:
        print("❌ QR processing needs fixes")

    if video_works:
        print("✅ Video creation is working")
    else:
        print("❌ Video creation has issues")

    print("\n💡 Check the saved debug images to see what the QR codes look like!")
    print("   Files saved: debug_qr_frame_*.png, debug_video_frame_*.png")


if __name__ == "__main__":
    main()