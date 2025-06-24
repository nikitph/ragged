import os

from ragged.services.uploader.interfaces import EncodedFileManager, FileUploader


class FileUploadService:
    """Orchestrates file uploads with dependency injection"""

    def __init__(self,
                 file_manager: EncodedFileManager,
                 uploader: FileUploader,
                 object_prefix: str = ""):
        self.file_manager = file_manager
        self.uploader = uploader
        self.object_prefix = object_prefix

    def execute_upload(self, base_file_path: str) -> None:
        """Main workflow execution"""
        for local_path in self.file_manager.get_file_paths(base_file_path):
            if not os.path.exists(local_path):
                print(f"Skipping missing file: {local_path}")
                continue

            object_key = self.object_prefix + os.path.basename(local_path)
            if self.uploader.upload(local_path, object_key):
                print(f"Uploaded {os.path.basename(local_path)} to {object_key}")