#!/usr/bin/env python3
"""
QR Processing Fix for Ragged Video System
Fixes the qr_to_frame function to preserve QR code quality

This script creates a fixed version of the utils.py file
"""

import shutil
from pathlib import Path


def create_fixed_qr_to_frame():
    """Create the fixed qr_to_frame function"""

    fixed_function = '''def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert QR PIL image to video frame with improved quality preservation

    Args:
        qr_image: PIL Image of QR code
        frame_size: Target frame size (width, height)

    Returns:
        OpenCV frame array (BGR format)
    """
    # Use the original size if it's smaller than target to avoid upscaling
    original_size = qr_image.size
    target_width, target_height = frame_size

    # Calculate the scaling factor to fit the QR code within the frame
    scale_x = target_width / original_size[0]
    scale_y = target_height / original_size[1]
    scale = min(scale_x, scale_y)

    # Calculate new size maintaining aspect ratio
    new_width = int(original_size[0] * scale)
    new_height = int(original_size[1] * scale)

    # Resize with high quality interpolation
    resized_qr = qr_image.resize((new_width, new_height), Image.Resampling.NEAREST)  # Use NEAREST for QR codes

    # Convert to RGB mode if necessary
    if resized_qr.mode != 'RGB':
        resized_qr = resized_qr.convert('RGB')

    # Create a white background frame
    frame_rgb = Image.new('RGB', frame_size, color='white')

    # Calculate position to center the QR code
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the QR code onto the center of the frame
    frame_rgb.paste(resized_qr, (x_offset, y_offset))

    # Convert to numpy array
    img_array = np.array(frame_rgb, dtype=np.uint8)

    # Convert from RGB to BGR (OpenCV format)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return frame'''

    return fixed_function


def backup_original_file():
    """Backup the original utils.py file"""
    utils_file = Path("ragged/video/utils.py")
    backup_file = Path("ragged/video/utils.py.backup")

    if utils_file.exists() and not backup_file.exists():
        shutil.copy2(utils_file, backup_file)
        print(f"✅ Backed up original file to: {backup_file}")
        return True
    elif backup_file.exists():
        print(f"ℹ️  Backup already exists: {backup_file}")
        return True
    else:
        print(f"❌ Could not find utils.py file")
        return False


def apply_fix():
    """Apply the QR processing fix"""

    utils_file = Path("ragged/video/utils.py")

    if not utils_file.exists():
        print(f"❌ Could not find {utils_file}")
        return False

    # Read the current file
    with open(utils_file, 'r') as f:
        content = f.read()

    # Find the qr_to_frame function and replace it
    import re

    # Pattern to match the qr_to_frame function
    pattern = r'def qr_to_frame\(.*?\n(?:    .*\n)*?    return frame'

    fixed_function = create_fixed_qr_to_frame()

    # Replace the function
    new_content = re.sub(pattern, fixed_function, content, flags=re.MULTILINE | re.DOTALL)

    if new_content != content:
        # Write the fixed version
        with open(utils_file, 'w') as f:
            f.write(new_content)
        print(f"✅ Applied QR processing fix to: {utils_file}")
        return True
    else:
        print(f"⚠️  Could not find qr_to_frame function to replace")
        return False


def create_manual_fix_instructions():
    """Create manual fix instructions if automatic patching fails"""

    instructions = """
# Manual QR Processing Fix Instructions

If the automatic fix didn't work, manually replace the `qr_to_frame` function in `ragged/video/utils.py`:

## 1. Open `ragged/video/utils.py`

## 2. Find the `qr_to_frame` function (around line 60-80)

## 3. Replace the entire function with this improved version:

```python
def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    \"\"\"
    Convert QR PIL image to video frame with improved quality preservation

    Args:
        qr_image: PIL Image of QR code
        frame_size: Target frame size (width, height)

    Returns:
        OpenCV frame array (BGR format)
    \"\"\"
    # Use the original size if it's smaller than target to avoid upscaling
    original_size = qr_image.size
    target_width, target_height = frame_size

    # Calculate the scaling factor to fit the QR code within the frame
    scale_x = target_width / original_size[0]
    scale_y = target_height / original_size[1]
    scale = min(scale_x, scale_y)

    # Calculate new size maintaining aspect ratio
    new_width = int(original_size[0] * scale)
    new_height = int(original_size[1] * scale)

    # Resize with high quality interpolation - use NEAREST for QR codes to preserve sharp edges
    resized_qr = qr_image.resize((new_width, new_height), Image.Resampling.NEAREST)

    # Convert to RGB mode if necessary
    if resized_qr.mode != 'RGB':
        resized_qr = resized_qr.convert('RGB')

    # Create a white background frame
    frame_rgb = Image.new('RGB', frame_size, color='white')

    # Calculate position to center the QR code
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the QR code onto the center of the frame
    frame_rgb.paste(resized_qr, (x_offset, y_offset))

    # Convert to numpy array
    img_array = np.array(frame_rgb, dtype=np.uint8)

    # Convert from RGB to BGR (OpenCV format)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return frame
```

## 4. Save the file

## 5. Test the fix:
```bash
python check_golden_path.py
```

The key changes:
- Use `Image.Resampling.NEAREST` instead of `LANCZOS` for QR codes (preserves sharp edges)
- Center the QR code on a white background 
- Maintain aspect ratio properly
- Avoid unnecessary upscaling
"""

    with open("QR_FIX_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)

    print(f"📝 Created manual fix instructions: QR_FIX_INSTRUCTIONS.md")


def test_fix():
    """Test if the fix worked"""
    print("\n🧪 Testing the fix...")

    try:
        from ragged.video.utils import encode_to_qr, decode_qr, qr_to_frame

        # Test the round-trip
        test_data = "QR fix test data"

        qr_image = encode_to_qr(test_data)
        frame = qr_to_frame(qr_image, (256, 256))
        decoded = decode_qr(frame)

        if decoded == test_data:
            print("✅ QR round-trip test PASSED!")
            print(f"   Successfully encoded and decoded: '{test_data}'")
            return True
        else:
            print(f"❌ QR round-trip test FAILED")
            print(f"   Expected: '{test_data}'")
            print(f"   Got: '{decoded}'")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main fix application function"""
    print("🔧 QR Processing Fix for Ragged Video System")
    print("=" * 60)

    # Step 1: Backup original file
    print("\n📋 Step 1: Backing up original file...")
    if not backup_original_file():
        print("❌ Could not backup original file")
        return 1

    # Step 2: Apply the fix
    print("\n🔧 Step 2: Applying QR processing fix...")
    if apply_fix():
        print("✅ Fix applied successfully")
    else:
        print("⚠️  Automatic fix failed, creating manual instructions...")
        create_manual_fix_instructions()
        return 1

    # Step 3: Test the fix
    print("\n🧪 Step 3: Testing the fix...")
    if test_fix():
        print("\n🎉 QR processing fix successful!")
        print("\n✅ Your golden path should now be fully working!")
        print("\nNext steps:")
        print("1. Run: python check_golden_path.py")
        print("2. Run your original tests to verify everything works")
        return 0
    else:
        print("\n⚠️  Fix didn't work as expected")
        print("Check QR_FIX_INSTRUCTIONS.md for manual fix steps")
        create_manual_fix_instructions()
        return 1


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)