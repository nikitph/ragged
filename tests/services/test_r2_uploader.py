import os
import unittest
import tempfile
import json
from ragged.services.uploader import R2Config, R2Uploader, FileManager, FileUploadService


class TestR2UploaderIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = R2Config(
            bucket=os.getenv("R2_TEST_BUCKET", "test-bucket"),
            endpoint=os.getenv("R2_ENDPOINT"),
            access_key=os.getenv("R2_ACCESS_KEY"),
            secret_key=os.getenv("R2_SECRET_KEY")
        )

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base_path = os.path.join(cls.temp_dir.name, "test_video.mp4")

        # Create test files
        with open(cls.base_path, 'wb') as f:
            f.write(b'mock mp4 data')
        with open(cls.base_path.replace('.mp4', '_manifest.json'), 'w') as f:
            json.dump({"test": "data"}, f)
        with open(cls.base_path.replace('.mp4', '_faiss.index'), 'wb') as f:
            f.write(b'mock index data')

    def test_upload_fidelity(self):
        uploader = R2Uploader(
            self.config.bucket,
            self.config.endpoint,
            self.config.access_key,
            self.config.secret_key
        )

        service = FileUploadService(
            FileManager(),
            uploader,
            "tests/"
        )

        # Verify bucket is empty before test
        for ext in ['', '_manifest.json', '_faiss.index']:
            object_key = f"tests/test_video{ext}"
            self.assertFalse(
                uploader.object_exists(object_key),
                f"File {object_key} already exists in bucket before test"
            )

        # Perform upload
        service.execute_upload(self.base_path)

        # Verify uploads
        for ext in ['', '_manifest.json', '_faiss.index']:
            object_key = f"tests/test_video{ext}"
            self.assertTrue(
                uploader.object_exists(object_key),
                f"File {object_key} not found in bucket after upload"
            )

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        uploader = R2Uploader(
            cls.config.bucket,
            cls.config.endpoint,
            cls.config.access_key,
            cls.config.secret_key
        )

        for ext in ['', '_manifest.json', '_faiss.index']:
            object_key = f"tests/test_video{ext}"
            uploader.delete_object(object_key)

        cls.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()