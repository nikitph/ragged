import os
from typing import List
import boto3
from botocore.config import Config

from ragged.config import settings
from ragged.services.uploader.file_manager import FileManager

from ragged.services.uploader.interfaces import EncodedFileManager, FileUploader
from ragged.services.uploader.upload_service import FileUploadService


class R2Uploader(FileUploader):
    """R2 Bucket upload implementation"""
    def __init__(self, bucket_name: str, endpoint_url: str, access_key: str, secret_key: str):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.boto_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(
                    region_name='auto',
                    signature_version='s3v4'
                )
            )

    def upload(self, local_path: str, object_key: str) -> bool:
        try:
            self.boto_client.upload_file(local_path, self.bucket_name, object_key)
            return True
        except Exception as e:
            print(f"Upload failed for {object_key}: {str(e)}")
            return False

    def object_exists(self, object_key: str) -> bool:
        try:
            self.boto_client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except:
            return False

    def delete_object(self, object_key: str) -> bool:
        try:
            self.boto_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            print(f"Successfully deleted {object_key}")
            return True
        except Exception as e:
            print(f"Failed to delete {object_key}: {str(e)}")
            return False

class R2Config:
    """Immutable configuration object"""
    def __init__(self, bucket: str, endpoint: str, access_key: str, secret_key: str):
        self.bucket = bucket
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

class UploadServiceBuilder:
    """Constructs service with dependencies"""
    @staticmethod
    def build( object_prefix: str = "") -> FileUploadService:
        config = R2Config(
            bucket=settings.R2_BUCKET,
            endpoint=settings.R2_ENDPOINT,
            access_key=settings.R2_ACCESS_KEY,
            secret_key=settings.R2_SECRET_KEY
        )
        uploader = R2Uploader(
            bucket_name=config.bucket,
            endpoint_url=config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key
        )
        file_manager = FileManager()
        return FileUploadService(file_manager, uploader, object_prefix)