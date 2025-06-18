"""
Video processing configuration for Ragged
Core codec settings, QR parameters, and processing defaults
"""

from typing import Dict, Any

# QR Code settings
QR_VERSION = 35  # 1-40, higher = more data capacity
QR_ERROR_CORRECTION = 'M'  # L, M, Q, H
QR_BOX_SIZE = 5
QR_BORDER = 3
QR_FILL_COLOR = "black"
QR_BACK_COLOR = "white"

# Chunking settings
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_OVERLAP = 32

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality
EMBEDDING_DIMENSION = 384

# Index settings
INDEX_TYPE = "Flat"  # Use Flat for reliability, IVF for scale
NLIST = 100  # Number of clusters for IVF index

# Retrieval settings
DEFAULT_TOP_K = 5
BATCH_SIZE = 100
MAX_WORKERS = 4
CACHE_SIZE = 1000

# Video codec configurations
MP4V_PARAMETERS = {
    "video_file_type": "mp4",
    "video_fps": 15,
    "frame_height": 256,
    "frame_width": 256,
    "video_crf": 18,
    "video_preset": "medium",
    "video_profile": "high",
    "pix_fmt": "yuv420p",
    "extra_ffmpeg_args": ""
}

H265_PARAMETERS = {
    "video_file_type": "mkv",
    "video_fps": 30,
    "video_crf": 28,
    "frame_height": 256,
    "frame_width": 256,
    "video_preset": "slower",
    "video_profile": "mainstillpicture",
    "pix_fmt": "yuv420p",
    "extra_ffmpeg_args": "-x265-params keyint=1:tune=stillimage"
}

H264_PARAMETERS = {
    "video_file_type": "mkv",
    "video_fps": 30,
    "video_crf": 28,
    "frame_height": 256,
    "frame_width": 256,
    "video_preset": "slower",
    "video_profile": "main",
    "pix_fmt": "yuv420p",
    "extra_ffmpeg_args": "-x264-params keyint=1:tune=stillimage"
}

# Codec mapping
CODEC_PARAMETERS = {
    "mp4v": MP4V_PARAMETERS,
    "h265": H265_PARAMETERS,
    "hevc": H265_PARAMETERS,
    "h264": H264_PARAMETERS,
    "avc": H264_PARAMETERS,
}

# Default video codec
VIDEO_CODEC = 'mp4v'  # Most compatible, no Docker required


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary"""
    return {
        "qr": {
            "version": QR_VERSION,
            "error_correction": QR_ERROR_CORRECTION,
            "box_size": QR_BOX_SIZE,
            "border": QR_BORDER,
            "fill_color": QR_FILL_COLOR,
            "back_color": QR_BACK_COLOR,
        },
        "codec": VIDEO_CODEC,
        "chunking": {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "overlap": DEFAULT_OVERLAP,
        },
        "retrieval": {
            "top_k": DEFAULT_TOP_K,
            "batch_size": BATCH_SIZE,
            "max_workers": MAX_WORKERS,
            "cache_size": CACHE_SIZE,
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
        },
        "index": {
            "type": INDEX_TYPE,
            "nlist": NLIST,
        }
    }


def get_codec_parameters(codec_name: str = None) -> Dict[str, Any]:
    """Get codec parameters for specified codec"""
    if codec_name is None:
        return CODEC_PARAMETERS

    if codec_name not in CODEC_PARAMETERS:
        raise ValueError(f"Unsupported codec: {codec_name}. Available: {list(CODEC_PARAMETERS.keys())}")

    return CODEC_PARAMETERS[codec_name]