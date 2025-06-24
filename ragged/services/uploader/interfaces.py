from abc import ABC, abstractmethod
import os
from typing import List

class FileUploader(ABC):
    """Abstract file uploader interface"""
    @abstractmethod
    def upload(self, local_path: str, object_key: str) -> bool:
        pass

class EncodedFileManager(ABC):
    """Abstract file management interface"""
    @abstractmethod
    def get_file_paths(self, base_path: str) -> List[str]:
        pass