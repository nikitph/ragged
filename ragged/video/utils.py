"""
Core utilities for QR code generation, decoding, and text processing
"""

import json
import base64
import gzip
import qrcode
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import logging
from functools import lru_cache

from .config import get_default_config

logger = logging.getLogger(__name__)


def encode_to_qr(data: str) -> Image.Image:
    """
    Encode data to QR code image with compression for large data

    Args:
        data: String data to encode

    Returns:
        PIL Image of QR code
    """
    config = get_default_config()["qr"]

    # Compress data if it's large (over 100 chars)
    if len(data) > 100:
        try:
            compressed = gzip.compress(data.encode())
            data = base64.b64encode(compressed).decode()
            data = "GZ:" + data  # Prefix to indicate compression
        except Exception as e:
            logger.warning(f"Failed to compress data: {e}")

    # Create QR code
    qr = qrcode.QRCode(
        version=config["version"],
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{config['error_correction']}"),
        box_size=config["box_size"],
        border=config["border"],
    )

    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=config["fill_color"], back_color=config["back_color"])
    return img


def decode_qr(image: np.ndarray) -> Optional[str]:
    """
    Decode QR code from image array

    Args:
        image: OpenCV image array (BGR format)

    Returns:
        Decoded string or None if decode fails
    """
    try:
        # Initialize OpenCV QR code detector
        detector = cv2.QRCodeDetector()

        # Detect and decode
        data, bbox, straight_qrcode = detector.detectAndDecode(image)

        if data:
            # Check if data was compressed
            if data.startswith("GZ:"):
                try:
                    compressed_data = base64.b64decode(data[3:])
                    data = gzip.decompress(compressed_data).decode()
                except Exception as e:
                    logger.warning(f"Failed to decompress QR data: {e}")
                    return None

            return data

    except Exception as e:
        logger.warning(f"QR decode failed: {e}")

    return None


def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert QR PIL image to video frame

    Args:
        qr_image: PIL Image of QR code
        frame_size: Target frame size (width, height)

    Returns:
        OpenCV frame array (BGR format)
    """
    # Resize to fit frame while maintaining aspect ratio
    qr_image = qr_image.resize(frame_size, Image.Resampling.LANCZOS)

    # Convert to RGB mode if necessary
    if qr_image.mode != 'RGB':
        qr_image = qr_image.convert('RGB')

    # Convert to numpy array
    img_array = np.array(qr_image, dtype=np.uint8)

    # Convert from RGB to BGR (OpenCV format)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return frame


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 32) -> List[str]:
    """
    Split text into overlapping chunks with smart sentence boundaries

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # If this is the last chunk, take everything
        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to break at sentence boundary
        chunk = text[start:end]

        # Look for sentence endings in the last 20% of the chunk
        sentence_break_start = int(len(chunk) * 0.8)
        sentence_endings = ['.', '!', '?', '\n\n']

        best_break = -1
        for ending in sentence_endings:
            pos = chunk.rfind(ending, sentence_break_start)
            if pos > best_break:
                best_break = pos

        # If we found a good break point, use it
        if best_break > 0:
            end = start + best_break + 1
            chunk = text[start:end].strip()
        else:
            # Fall back to word boundary
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:
                end = start + last_space
                chunk = text[start:end].strip()
            else:
                chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - overlap

        # Ensure we don't get stuck in infinite loop
        if next_start <= start:  # ✅ Comparing int with int
            start = start + 1  # Move forward at least one character
        else:
            start = next_start

    return chunks


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract single frame from video file

    Args:
        video_path: Path to video file
        frame_number: Frame index to extract (0-based)

    Returns:
        OpenCV frame array or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)
    try:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret and frame is not None:
            return frame
        else:
            logger.warning(f"Failed to extract frame {frame_number} from {video_path}")
            return None

    except Exception as e:
        logger.error(f"Error extracting frame {frame_number}: {e}")
        return None
    finally:
        cap.release()


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    """
    Extract and decode frame with LRU caching for performance

    Args:
        video_path: Path to video file
        frame_number: Frame index to extract

    Returns:
        Decoded text content or None
    """
    frame = extract_frame(video_path, frame_number)
    if frame is not None:
        return decode_qr(frame)
    return None


def validate_chunk_text(text: str) -> bool:
    """
    Validate that text chunk is suitable for QR encoding

    Args:
        text: Text chunk to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(text, str):
        return False

    text = text.strip()

    # Basic checks
    if len(text) == 0:
        return False

    if len(text) > 8192:  # QR code practical limit
        return False

    # Check if it's valid UTF-8
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False


def format_chunk_data(chunk_id: int, text: str, metadata: dict = None) -> str:
    """
    Format chunk data as JSON for QR encoding

    Args:
        chunk_id: Unique chunk identifier
        text: Chunk text content
        metadata: Optional metadata dictionary

    Returns:
        JSON formatted string
    """
    chunk_data = {
        "id": chunk_id,
        "text": text,
        "frame": chunk_id  # For compatibility
    }

    if metadata:
        chunk_data["metadata"] = metadata

    return json.dumps(chunk_data, ensure_ascii=False)