"""
Video Retriever - Fast semantic search and QR frame extraction
Handles video-based knowledge retrieval with FAISS search and QR decoding
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import cv2

from .utils import extract_frame, decode_qr, extract_and_decode_cached
from .index import IndexManager
from .config import get_default_config

logger = logging.getLogger(__name__)


class VideoRetriever:
    """
    Fast retrieval from QR code videos using semantic search
    """

    def __init__(self, video_file: str, index_file: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize VideoRetriever

        Args:
            video_file: Path to QR code video
            index_file: Path to index JSON file
            config: Optional configuration
        """
        self.video_file = str(Path(video_file).absolute())
        self.index_file = str(Path(index_file).absolute())
        self.config = config or get_default_config()

        # Load index
        self.index_manager = IndexManager(self.config)
        index_base = str(Path(index_file).with_suffix(''))
        self.index_manager.load(index_base)

        # Cache for decoded frames
        self._frame_cache = {}
        self._cache_size = self.config["retrieval"]["cache_size"]

        # Verify video file
        self._verify_video()

        logger.info(f"Initialized retriever with {self.get_stats()['index_stats']['total_chunks']} chunks")

    def _verify_video(self):
        """Verify video file is accessible and get basic info"""
        if not Path(self.video_file).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_file}")

        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")

        try:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)

            if self.total_frames <= 0:
                raise ValueError(f"Video has no frames: {self.video_file}")

            logger.info(f"Video has {self.total_frames} frames at {self.fps} fps")

        finally:
            cap.release()

    def search(self, query: str, top_k: int = None) -> List[str]:
        """
        Search for relevant chunks using semantic search

        Args:
            query: Search query
            top_k: Number of results to return (default from config)

        Returns:
            List of relevant text chunks
        """
        top_k = top_k or self.config["retrieval"]["top_k"]
        start_time = time.time()

        # Semantic search in index
        search_results = self.index_manager.search(query, top_k)

        if not search_results:
            logger.info(f"No results found for query: '{query}'")
            return []

        # Extract frame numbers and decode
        frame_numbers = [result[2]["frame"] for result in search_results]
        decoded_frames = self._decode_frames_parallel(frame_numbers)

        # Extract text from decoded data
        results = []
        for chunk_id, distance, metadata in search_results:
            frame_num = metadata["frame"]

            if frame_num in decoded_frames:
                try:
                    chunk_data = json.loads(decoded_frames[frame_num])
                    results.append(chunk_data["text"])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse QR data for frame {frame_num}: {e}")
                    # Fallback to metadata preview
                    results.append(metadata["text"])
            else:
                # Fallback to metadata preview if decode failed
                results.append(metadata["text"])

        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.3f}s for query: '{query[:50]}...'")

        return results

    def search_with_metadata(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search with full metadata including scores and frame numbers

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of result dictionaries with text, score, and metadata
        """
        top_k = top_k or self.config["retrieval"]["top_k"]
        start_time = time.time()

        # Semantic search
        search_results = self.index_manager.search(query, top_k)

        if not search_results:
            return []

        # Extract frame numbers and decode
        frame_numbers = [result[2]["frame"] for result in search_results]
        decoded_frames = self._decode_frames_parallel(frame_numbers)

        # Build detailed results
        results = []
        for chunk_id, distance, metadata in search_results:
            frame_num = metadata["frame"]

            # Get full text from decoded frame or use metadata preview
            if frame_num in decoded_frames:
                try:
                    chunk_data = json.loads(decoded_frames[frame_num])
                    text = chunk_data["text"]
                except (json.JSONDecodeError, KeyError):
                    text = metadata["text"]
            else:
                text = metadata["text"]

            # Convert distance to similarity score (higher = more similar)
            similarity_score = 1.0 / (1.0 + distance)

            result = {
                "text": text,
                "score": similarity_score,
                "chunk_id": chunk_id,
                "frame": frame_num,
                "metadata": metadata
            }
            results.append(result)

        elapsed = time.time() - start_time
        logger.info(f"Search with metadata completed in {elapsed:.3f}s")

        return results

    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """
        Get specific chunk by ID

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk text or None if not found
        """
        metadata = self.index_manager.get_chunk_by_id(chunk_id)
        if not metadata:
            return None

        frame_num = metadata["frame"]
        decoded = self._decode_single_frame(frame_num)

        if decoded:
            try:
                chunk_data = json.loads(decoded)
                return chunk_data["text"]
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback to metadata preview
        return metadata["text"]

    def get_chunks_by_frame(self, frame_number: int) -> List[str]:
        """
        Get all chunks from a specific frame

        Args:
            frame_number: Frame number to retrieve

        Returns:
            List of chunk texts from that frame
        """
        chunks_metadata = self.index_manager.get_chunks_by_frame(frame_number)

        if not chunks_metadata:
            return []

        # Decode the frame once
        decoded = self._decode_single_frame(frame_number)

        if decoded:
            try:
                chunk_data = json.loads(decoded)
                return [chunk_data["text"]]
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback to metadata previews
        return [metadata["text"] for metadata in chunks_metadata]

    def _decode_single_frame(self, frame_number: int) -> Optional[str]:
        """Decode single frame with caching"""
        # Check cache first
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]

        # Decode frame
        result = extract_and_decode_cached(self.video_file, frame_number)

        # Update cache if successful and not full
        if result and len(self._frame_cache) < self._cache_size:
            self._frame_cache[frame_number] = result

        return result

    def _decode_frames_parallel(self, frame_numbers: List[int]) -> Dict[int, str]:
        """
        Decode multiple frames in parallel with caching

        Args:
            frame_numbers: List of frame numbers to decode

        Returns:
            Dict mapping frame number to decoded data
        """
        # Check cache first
        results = {}
        uncached_frames = []

        for frame_num in frame_numbers:
            if frame_num in self._frame_cache:
                results[frame_num] = self._frame_cache[frame_num]
            else:
                uncached_frames.append(frame_num)

        if not uncached_frames:
            return results

        # Decode uncached frames in parallel
        max_workers = min(self.config["retrieval"]["max_workers"], len(uncached_frames))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit decode tasks
            future_to_frame = {
                executor.submit(extract_and_decode_cached, self.video_file, frame_num): frame_num
                for frame_num in uncached_frames
            }

            # Collect results
            for future in future_to_frame:
                frame_num = future_to_frame[future]
                try:
                    decoded_data = future.result(timeout=10)  # 10 second timeout per frame
                    if decoded_data:
                        results[frame_num] = decoded_data
                        # Update cache if not full
                        if len(self._frame_cache) < self._cache_size:
                            self._frame_cache[frame_num] = decoded_data
                except Exception as e:
                    logger.warning(f"Failed to decode frame {frame_num}: {e}")

        return results

    def get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]:
        """
        Get chunk with surrounding context

        Args:
            chunk_id: Central chunk ID
            window_size: Number of chunks before/after to include

        Returns:
            List of chunks in context window
        """
        chunks = []

        for offset in range(-window_size, window_size + 1):
            target_id = chunk_id + offset
            chunk = self.get_chunk_by_id(target_id)
            if chunk:
                chunks.append(chunk)

        return chunks

    def prefetch_frames(self, frame_numbers: List[int]):
        """
        Prefetch frames into cache for faster retrieval

        Args:
            frame_numbers: List of frame numbers to prefetch
        """
        to_prefetch = [f for f in frame_numbers if f not in self._frame_cache]

        if to_prefetch:
            logger.info(f"Prefetching {len(to_prefetch)} frames...")
            decoded = self._decode_frames_parallel(to_prefetch)
            logger.info(f"Prefetched {len(decoded)} frames successfully")

    def clear_cache(self):
        """Clear frame cache"""
        self._frame_cache.clear()
        extract_and_decode_cached.cache_clear()
        logger.info("Cleared frame cache")

    def get_video_info(self) -> Dict[str, Any]:
        """Get video file information"""
        video_path = Path(self.video_file)
        file_size_mb = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0

        return {
            "video_file": self.video_file,
            "file_size_mb": file_size_mb,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "video_info": self.get_video_info(),
            "cache_size": len(self._frame_cache),
            "max_cache_size": self._cache_size,
            "index_stats": self.index_manager.get_stats()
        }

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate video and index integrity

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating video and index integrity...")

        issues = []
        stats = {"total_frames": 0, "readable_frames": 0, "valid_qr_frames": 0}

        # Test a sample of frames
        total_chunks = self.index_manager.get_stats()["total_chunks"]
        sample_size = min(100, total_chunks)  # Test up to 100 frames
        sample_frames = list(range(0, total_chunks, max(1, total_chunks // sample_size)))

        for frame_num in sample_frames:
            stats["total_frames"] += 1

            # Try to extract frame
            frame = extract_frame(self.video_file, frame_num)
            if frame is not None:
                stats["readable_frames"] += 1

                # Try to decode QR
                decoded = decode_qr(frame)
                if decoded:
                    stats["valid_qr_frames"] += 1
                    try:
                        json.loads(decoded)  # Validate JSON
                    except json.JSONDecodeError:
                        issues.append(f"Frame {frame_num}: Invalid JSON in QR code")
                else:
                    issues.append(f"Frame {frame_num}: Could not decode QR code")
            else:
                issues.append(f"Frame {frame_num}: Could not extract frame")

        # Calculate integrity percentage
        integrity_percent = (stats["valid_qr_frames"] / stats["total_frames"] * 100) if stats["total_frames"] > 0 else 0

        result = {
            "integrity_percent": integrity_percent,
            "stats": stats,
            "issues": issues,
            "is_healthy": integrity_percent >= 95  # Consider healthy if 95%+ frames are valid
        }

        logger.info(f"Integrity check: {integrity_percent:.1f}% valid frames")

        return result