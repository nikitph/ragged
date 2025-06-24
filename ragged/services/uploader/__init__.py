"""
Uploader Service Package

Provides abstractions and implementations for uploading files to cloud storage.
Designed with SOLID principles for easy extensibility.

Key Components:
- FileUploader: Abstract base class for upload implementations
- EncodedFileManager: Abstract base class for file collection
- R2Uploader: Cloudflare R2 implementation
- MP4EncoderFileManager: File collector for MP4 encoder outputs
- FileUploadService: Main orchestration service
- UploadServiceBuilder: Dependency injection helper
"""

from .interfaces import FileUploader, EncodedFileManager
from .r2_uploader import R2Config, R2Uploader, UploadServiceBuilder
from .file_manager import FileManager
from .upload_service import FileUploadService

__all__ = [
    # Interfaces
    'FileUploader',
    'EncodedFileManager',

    # Implementations
    'R2Config',
    'R2Uploader',
    'FileManager',

    # Services
    'FileUploadService',
    'UploadServiceBuilder'
]

# Package version
__version__ = "1.0.0"