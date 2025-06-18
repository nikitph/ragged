"""
Video Memory Encoder - Creates QR code videos from text chunks
Handles text processing, QR generation, video creation, and index building
"""

import json
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from tqdm import tqdm

from .utils import chunk_text, encode_to_qr, qr_to_frame, format_chunk_data, validate_chunk_text
from .index import IndexManager
from .config import get_default_config, get_codec_parameters, VIDEO_CODEC

logger = logging.getLogger(__name__)


class VideoEncoder:
    """
    Encodes text chunks into searchable QR code videos
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize VideoEncoder

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()
        self.chunks = []
        self.index_manager = IndexManager(self.config)

        logger.info("VideoEncoder initialized")

    def add_chunks(self, chunks: List[str]):
        """
        Add pre-chunked text pieces to be encoded

        Args:
            chunks: List of text chunks
        """
        valid_chunks = [chunk for chunk in chunks if validate_chunk_text(chunk)]
        invalid_count = len(chunks) - len(valid_chunks)

        self.chunks.extend(valid_chunks)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid chunks")

        logger.info(f"Added {len(valid_chunks)} chunks. Total: {len(self.chunks)}")

    def add_text(self, text: str, chunk_size: int = None, overlap: int = None, metadata: dict = None):
        """
        Add text with automatic chunking

        Args:
            text: Text to chunk and add
            chunk_size: Override default chunk size
            overlap: Override default overlap
            metadata: Optional metadata to attach to chunks
        """
        chunk_size = chunk_size or self.config["chunking"]["chunk_size"]
        overlap = overlap or self.config["chunking"]["overlap"]

        chunks = chunk_text(text, chunk_size, overlap)

        # Add metadata to chunks if provided
        if metadata:
            # For now, just add chunks directly
            # TODO: In future, store metadata with chunks
            pass

        self.add_chunks(chunks)

    def add_file(self, file_path: str, chunk_size: int = None, overlap: int = None):
        """
        Add text file content

        Args:
            file_path: Path to text file
            chunk_size: Override default chunk size
            overlap: Override default overlap
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            metadata = {"source": os.path.basename(file_path)}
            self.add_text(text, chunk_size, overlap, metadata)

            logger.info(f"Added content from file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

    def build_video(self, output_video: str, output_index: str,
                   codec: str = None, show_progress: bool = True) -> Dict[str, Any]:
        """
        Build QR code video and search index from chunks

        Args:
            output_video: Path to output video file
            output_index: Path to output index file
            codec: Video codec to use (default from config)
            show_progress: Show progress bars

        Returns:
            Dictionary with build statistics
        """
        if not self.chunks:
            raise ValueError("No chunks to encode. Use add_chunks() or add_text() first.")

        codec = codec or self.config["codec"]
        output_video_path = Path(output_video)
        output_index_path = Path(output_index)

        # Ensure output directories exist
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building video with {len(self.chunks)} chunks using {codec} codec")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate QR frames
            frames_dir = self._generate_qr_frames(temp_path, show_progress)

            # Create video
            if codec == "mp4v":
                stats = self._encode_with_opencv(frames_dir, output_video_path, codec, show_progress)
            else:
                stats = self._encode_with_ffmpeg(frames_dir, output_video_path, codec, show_progress)

            # Build search index
            if show_progress:
                logger.info("Building search index...")

            frame_numbers = list(range(len(self.chunks)))
            self.index_manager.add_chunks(self.chunks, frame_numbers, show_progress)

            # Save index
            index_base = str(output_index_path.with_suffix(''))
            self.index_manager.save(index_base)

            # Compile final statistics
            stats.update({
                "total_chunks": len(self.chunks),
                "video_file": str(output_video_path),
                "index_file": str(output_index_path),
                "index_stats": self.index_manager.get_stats()
            })

            if show_progress:
                logger.info(f"Successfully built video: {output_video_path}")
                logger.info(f"Video duration: {stats.get('duration_seconds', 0):.1f} seconds")
                logger.info(f"Video size: {stats.get('video_size_mb', 0):.1f} MB")

            return stats

    def _generate_qr_frames(self, temp_dir: Path, show_progress: bool) -> Path:
        """Generate QR code frames to temporary directory"""
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        chunks_iter = enumerate(self.chunks)
        if show_progress:
            chunks_iter = tqdm(chunks_iter, total=len(self.chunks), desc="Generating QR frames")

        for frame_num, chunk in chunks_iter:
            # Format chunk data as JSON
            chunk_data = format_chunk_data(frame_num, chunk)

            # Generate QR code
            qr_image = encode_to_qr(chunk_data)

            # Save as PNG frame
            frame_path = frames_dir / f"frame_{frame_num:06d}.png"
            qr_image.save(frame_path)

        logger.info(f"Generated {len(self.chunks)} QR frames in {frames_dir}")
        return frames_dir

    def _encode_with_opencv(self, frames_dir: Path, output_file: Path, codec: str,
                           show_progress: bool) -> Dict[str, Any]:
        """Encode video using OpenCV (for mp4v codec)"""
        codec_config = get_codec_parameters(codec)

        if show_progress:
            logger.info(f"Encoding with OpenCV using {codec} codec...")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = codec_config["video_fps"]
        frame_size = (codec_config["frame_width"], codec_config["frame_height"])

        writer = cv2.VideoWriter(str(output_file), fourcc, fps, frame_size)

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_file}")

        try:
            # Get frame files
            frame_files = sorted(frames_dir.glob("frame_*.png"))

            frame_iter = enumerate(frame_files)
            if show_progress:
                frame_iter = tqdm(frame_iter, total=len(frame_files), desc="Writing video frames")

            frame_count = 0
            for _, frame_file in frame_iter:
                # Load frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Could not load frame: {frame_file}")
                    continue

                # Resize if needed
                if frame.shape[:2][::-1] != frame_size:
                    frame = cv2.resize(frame, frame_size)

                # Write frame
                writer.write(frame)
                frame_count += 1

            # Calculate stats
            file_size_mb = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
            duration_seconds = frame_count / fps if fps > 0 else 0

            return {
                "backend": "opencv",
                "codec": codec,
                "total_frames": frame_count,
                "video_size_mb": file_size_mb,
                "fps": fps,
                "duration_seconds": duration_seconds
            }

        finally:
            writer.release()

    def _encode_with_ffmpeg(self, frames_dir: Path, output_file: Path, codec: str,
                           show_progress: bool) -> Dict[str, Any]:
        """Encode video using FFmpeg (for advanced codecs)"""
        codec_config = get_codec_parameters(codec)

        if show_progress:
            logger.info(f"Encoding with FFmpeg using {codec} codec...")

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(frames_dir, output_file, codec)

        # Execute FFmpeg
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

            # Calculate stats
            frame_count = len(list(frames_dir.glob("frame_*.png")))
            file_size_mb = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
            fps = codec_config["video_fps"]
            duration_seconds = frame_count / fps if fps > 0 else 0

            return {
                "backend": "ffmpeg",
                "codec": codec,
                "total_frames": frame_count,
                "video_size_mb": file_size_mb,
                "fps": fps,
                "duration_seconds": duration_seconds
            }

        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg encoding timed out")
        except Exception as e:
            raise RuntimeError(f"FFmpeg execution error: {e}")

    def _build_ffmpeg_command(self, frames_dir: Path, output_file: Path, codec: str) -> List[str]:
        """Build FFmpeg command for video encoding"""
        codec_config = get_codec_parameters(codec)

        # FFmpeg codec mapping
        ffmpeg_codec_map = {
            "h265": "libx265", "hevc": "libx265",
            "h264": "libx264", "avc": "libx264",
        }

        ffmpeg_codec = ffmpeg_codec_map.get(codec, codec)

        # Build command
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(codec_config["video_fps"]),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', ffmpeg_codec,
            '-preset', codec_config["video_preset"],
            '-crf', str(codec_config["video_crf"]),
            '-pix_fmt', codec_config["pix_fmt"],
        ]

        # Add scaling if needed
        target_width = codec_config["frame_width"]
        target_height = codec_config["frame_height"]
        cmd.extend(['-vf', f'scale={target_width}:{target_height}'])

        # Add profile if specified
        if codec_config.get("video_profile"):
            cmd.extend(['-profile:v', codec_config["video_profile"]])

        # Add extra parameters
        if codec_config.get("extra_ffmpeg_args"):
            extra_args = codec_config["extra_ffmpeg_args"].split()
            cmd.extend(extra_args)

        # Streaming optimizations
        cmd.extend(['-movflags', '+faststart'])

        # Output file
        cmd.append(str(output_file))

        return cmd

    def clear(self):
        """Clear all chunks and reset encoder"""
        self.chunks = []
        self.index_manager = IndexManager(self.config)
        logger.info("Cleared all chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        total_chars = sum(len(chunk) for chunk in self.chunks) if self.chunks else 0
        avg_chunk_size = total_chars / len(self.chunks) if self.chunks else 0

        return {
            "total_chunks": len(self.chunks),
            "total_characters": total_chars,
            "avg_chunk_size": avg_chunk_size,
            "config": self.config
        }

    @classmethod
    def from_text(cls, text: str, chunk_size: int = None, overlap: int = None,
                  config: Optional[Dict[str, Any]] = None) -> 'VideoEncoder':
        """
        Create encoder from text with automatic chunking

        Args:
            text: Text to encode
            chunk_size: Override default chunk size
            overlap: Override default overlap
            config: Optional configuration

        Returns:
            VideoEncoder instance with chunks loaded
        """
        encoder = cls(config)
        encoder.add_text(text, chunk_size, overlap)
        return encoder

    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = None, overlap: int = None,
                  config: Optional[Dict[str, Any]] = None) -> 'VideoEncoder':
        """
        Create encoder from text file

        Args:
            file_path: Path to text file
            chunk_size: Override default chunk size
            overlap: Override default overlap
            config: Optional configuration

        Returns:
            VideoEncoder instance with chunks loaded
        """
        encoder = cls(config)
        encoder.add_file(file_path, chunk_size, overlap)
        return encoder