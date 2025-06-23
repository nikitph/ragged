import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import hashlib
import os
import faiss
import requests
import time
import pickle
import tempfile
from functools import lru_cache
from urllib.parse import urlparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import io


class CDNVectorMP4Decoder:
    """Enhanced decoder with full CDN support, range requests, and intelligent caching"""

    def __init__(self,
                 mp4_path: str,
                 manifest_path: Optional[str] = None,
                 faiss_path: Optional[str] = None,
                 cache_size: int = 100,
                 disk_cache_dir: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 chunk_size: int = 8192,
                 enable_prefetching: bool = True):
        """
        Initialize enhanced decoder with full CDN capabilities

        Args:
            mp4_path: Path or URL to MP4 file
            manifest_path: Path or URL to manifest file (optional)
            faiss_path: Path or URL to Faiss index file (optional)
            cache_size: Maximum number of fragments to cache in memory
            disk_cache_dir: Directory for persistent disk cache (optional)
            max_retries: Maximum number of retry attempts for downloads
            timeout: Request timeout in seconds
            chunk_size: Size of chunks for streaming downloads
            enable_prefetching: Whether to prefetch adjacent fragments
        """
        self.mp4_path = mp4_path
        self.manifest_path = manifest_path or self._get_companion_path(mp4_path, '_manifest.json')
        self.faiss_path = faiss_path or self._get_companion_path(mp4_path, '_faiss.index')

        self.cache_size = cache_size
        self.disk_cache_dir = disk_cache_dir
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.enable_prefetching = enable_prefetching

        # Initialize caches and state
        self.memory_cache = {}
        self.manifest = None
        self.faiss_index = None
        self.file_sizes = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prefetch_futures = {}

        # Determine if files are remote
        self.is_mp4_remote = self._is_url(mp4_path)
        self.is_manifest_remote = self._is_url(self.manifest_path)
        self.is_faiss_remote = self._is_url(self.faiss_path)

        # Setup disk cache
        if self.disk_cache_dir:
            os.makedirs(self.disk_cache_dir, exist_ok=True)

        # Initialize components
        self._load_manifest()
        self._load_faiss_index()

    def _is_url(self, path: str) -> bool:
        """Check if path is a URL"""
        return urlparse(path).scheme in ['http', 'https']

    def _get_companion_path(self, base_path: str, suffix: str) -> str:
        """Generate companion file path for manifest or faiss index"""
        if self._is_url(base_path):
            # For URLs, replace extension
            base_url = base_path.rsplit('.', 1)[0] if '.' in base_path else base_path
            return base_url + suffix
        else:
            # For local files, replace extension
            return base_path.replace('.mp4', suffix)

    def _get_file_size(self, path: str) -> int:
        """Get file size for local or remote files"""
        if path in self.file_sizes:
            return self.file_sizes[path]

        if self._is_url(path):
            # Use HEAD request to get file size
            try:
                response = requests.head(path, timeout=self.timeout)
                size = int(response.headers.get('Content-Length', 0))
                self.file_sizes[path] = size
                return size
            except:
                return 0
        else:
            # Local file
            try:
                size = os.path.getsize(path)
                self.file_sizes[path] = size
                return size
            except:
                return 0

    def _supports_range_requests(self, url: str) -> bool:
        """Check if server supports HTTP range requests"""
        if not self._is_url(url):
            return True  # Local files always support "ranges"

        try:
            # Send a small range request to test
            headers = {'Range': 'bytes=0-0'}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            return response.status_code == 206
        except:
            return False

    def _download_with_retries(self, url: str, headers: Optional[Dict] = None) -> bytes:
        """Download data with retry logic and exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to download after {self.max_retries} attempts: {e}")
                wait_time = (2 ** attempt) + (time.time() % 1)  # Add jitter
                print(f"Download attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

    def _download_range(self, path: str, start_byte: int, end_byte: int) -> bytes:
        """Download specific byte range from local or remote file"""
        if not self._is_url(path):
            # Local file
            with open(path, 'rb') as f:
                f.seek(start_byte)
                return f.read(end_byte - start_byte + 1)
        else:
            # Remote file - use range request
            headers = {'Range': f'bytes={start_byte}-{end_byte}'}
            return self._download_with_retries(path, headers)

    def _stream_download(self, url: str, total_size: Optional[int] = None) -> bytes:
        """Stream download with progress tracking"""
        if not self._is_url(url):
            with open(url, 'rb') as f:
                return f.read()

        data = io.BytesIO()
        downloaded = 0

        try:
            with requests.get(url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        data.write(chunk)
                        downloaded += len(chunk)

                        # Optional: Progress callback
                        if total_size and downloaded % (self.chunk_size * 10) == 0:
                            progress = (downloaded / total_size) * 100
                            print(f"Downloaded {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)")

            return data.getvalue()
        except Exception as e:
            raise RuntimeError(f"Failed to stream download {url}: {e}")

    def _load_manifest(self):
        """Load manifest from local file or CDN"""
        cache_key = f"manifest_{hashlib.md5(self.manifest_path.encode()).hexdigest()}"

        # Try disk cache first
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        self.manifest = json.load(f)
                        print(f"Loaded manifest from disk cache")
                        return
                except:
                    pass

        # Download or load manifest
        try:
            if self.is_manifest_remote:
                print(f"Downloading manifest from {self.manifest_path}")
                manifest_data = self._download_with_retries(self.manifest_path)
                self.manifest = json.loads(manifest_data.decode('utf-8'))
            else:
                if os.path.exists(self.manifest_path):
                    with open(self.manifest_path, 'r') as f:
                        self.manifest = json.load(f)
                else:
                    # Try to extract from MP4
                    self.manifest = self._extract_manifest_from_mp4()

            # Cache to disk
            if self.disk_cache_dir and self.manifest:
                cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.json")
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(self.manifest, f)
                except:
                    pass

            print(f"Loaded manifest with {self.manifest['metadata']['total_vectors']} vectors")

        except Exception as e:
            raise RuntimeError(f"Failed to load manifest: {e}")

    def _load_faiss_index(self):
        """Load Faiss index from local file or CDN with caching"""
        if not self.faiss_path:
            print("No Faiss index path provided")
            return

        cache_key = f"faiss_{hashlib.md5(self.faiss_path.encode()).hexdigest()}"

        # Try disk cache first
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.index")
            if os.path.exists(cache_file):
                try:
                    self.faiss_index = faiss.read_index(cache_file)
                    print(f"Loaded Faiss index from disk cache")
                    return
                except:
                    pass

        try:
            if self.is_faiss_remote:
                print(f"Downloading Faiss index from {self.faiss_path}")
                index_size = self._get_file_size(self.faiss_path)
                index_data = self._stream_download(self.faiss_path, index_size)

                # Save to temporary file and load
                with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp_file:
                    tmp_file.write(index_data)
                    tmp_path = tmp_file.name

                self.faiss_index = faiss.read_index(tmp_path)
                os.unlink(tmp_path)
            else:
                if os.path.exists(self.faiss_path):
                    self.faiss_index = faiss.read_index(self.faiss_path)
                else:
                    print(f"Faiss index not found at {self.faiss_path}")
                    return

            # Cache to disk
            if self.disk_cache_dir and self.faiss_index:
                cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.index")
                try:
                    faiss.write_index(self.faiss_index, cache_file)
                except:
                    pass

            print(f"Loaded Faiss index with {self.faiss_index.ntotal} vectors")

        except Exception as e:
            print(f"Warning: Failed to load Faiss index: {e}")

    def _extract_manifest_from_mp4(self) -> Dict:
        """Extract manifest from MP4 file with range request support"""
        if self.is_mp4_remote:
            # For remote files, download first 1MB to search for manifest
            print("Searching for manifest in remote MP4 file...")
            header_data = self._download_range(self.mp4_path, 0, 1024 * 1024 - 1)
            return self._parse_manifest_from_data(header_data)
        else:
            # Local file
            with open(self.mp4_path, 'rb') as f:
                data = f.read(1024 * 1024)  # Read first 1MB
                return self._parse_manifest_from_data(data)

    def _parse_manifest_from_data(self, data: bytes) -> Dict:
        """Parse manifest from MP4 data"""
        offset = 0
        while offset < len(data) - 8:
            try:
                box_size, box_type = struct.unpack('>I4s', data[offset:offset + 8])
                box_type = box_type.decode('ascii')

                if box_type == 'manf':  # Manifest box
                    manifest_data = data[offset + 8:offset + box_size]
                    return json.loads(manifest_data.decode('utf-8'))

                offset += box_size
                if box_size == 0:  # Avoid infinite loop
                    break
            except:
                offset += 1

        raise ValueError("No manifest found in MP4 file")

    def _get_cache_key(self, fragment_id: int) -> str:
        """Generate cache key for fragment"""
        return f"fragment_{fragment_id}_{hashlib.md5(self.mp4_path.encode()).hexdigest()[:8]}"

    def _get_fragment_from_cache(self, fragment_id: int) -> Optional[Dict]:
        """Try to get fragment from memory or disk cache"""
        cache_key = self._get_cache_key(fragment_id)

        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # Check disk cache
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        fragment = pickle.load(f)
                        # Add to memory cache
                        self._add_to_memory_cache(cache_key, fragment)
                        return fragment
                except:
                    pass

        return None

    def _add_to_memory_cache(self, cache_key: str, fragment: Dict):
        """Add fragment to memory cache with LRU eviction"""
        # Simple FIFO eviction when cache is full
        while len(self.memory_cache) >= self.cache_size:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        self.memory_cache[cache_key] = fragment

    def _save_fragment_to_cache(self, fragment_id: int, fragment: Dict):
        """Save fragment to both memory and disk cache"""
        cache_key = self._get_cache_key(fragment_id)

        # Add to memory cache
        self._add_to_memory_cache(cache_key, fragment)

        # Save to disk cache
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(fragment, f)
            except Exception as e:
                print(f"Failed to save fragment to disk cache: {e}")

    def _download_fragment(self, fragment_id: int) -> Dict:
        """Download and parse a specific fragment"""
        if not self.manifest:
            raise RuntimeError("Manifest not loaded")

        # Find fragment info
        fragment_info = None
        for frag in self.manifest["metadata"]["fragments"]:
            if frag["id"] == fragment_id:
                fragment_info = frag
                break

        if not fragment_info:
            raise ValueError(f"Fragment {fragment_id} not found in manifest")

        # Download fragment data using range request
        start_byte = fragment_info["byte_start"]
        end_byte = fragment_info["byte_end"] - 1  # HTTP ranges are inclusive

        print(f"Downloading fragment {fragment_id} (bytes {start_byte}-{end_byte})")
        fragment_data = self._download_range(self.mp4_path, start_byte, end_byte)

        # Parse the fragment
        return self._parse_fragment_data(fragment_data, fragment_info)

    def _parse_fragment_data(self, data: bytes, fragment_info: Dict) -> Dict:
        """Parse binary fragment data"""
        # Skip moof box to get to mdat content
        mdat_offset = fragment_info.get("mdat_start", 8) - fragment_info.get("byte_start", 0)
        mdat_data = data[mdat_offset + 8:]  # Skip mdat header

        offset = 0

        # Read header
        header_size = struct.unpack('<I', mdat_data[offset:offset + 4])[0]
        offset += 4
        header_json = mdat_data[offset:offset + header_size].decode('utf-8')
        header = json.loads(header_json)
        offset += header_size

        # Read vectors
        vectors_size = struct.unpack('<I', mdat_data[offset:offset + 4])[0]
        offset += 4
        vectors_data = mdat_data[offset:offset + vectors_size]
        vectors = np.frombuffer(vectors_data, dtype=np.float32).reshape(-1, header['vector_dim'])
        offset += vectors_size

        # Read metadata
        metadata_size = struct.unpack('<I', mdat_data[offset:offset + 4])[0]
        offset += 4
        metadata_json = mdat_data[offset:offset + metadata_size].decode('utf-8')
        metadata = json.loads(metadata_json)

        return {
            "header": header,
            "vectors": vectors,
            "metadata": metadata
        }

    def _prefetch_adjacent_fragments(self, fragment_id: int):
        """Prefetch adjacent fragments in background"""
        if not self.enable_prefetching:
            return

        # Prefetch next fragment
        next_fragment_id = fragment_id + 1
        if next_fragment_id not in self.prefetch_futures:
            if any(f["id"] == next_fragment_id for f in self.manifest["metadata"]["fragments"]):
                future = self.executor.submit(self._get_fragment, next_fragment_id)
                self.prefetch_futures[next_fragment_id] = future

    def _get_fragment(self, fragment_id: int) -> Dict:
        """Get fragment with caching and prefetching"""
        # Check cache first
        fragment = self._get_fragment_from_cache(fragment_id)
        if fragment:
            # Start prefetching adjacent fragments
            self._prefetch_adjacent_fragments(fragment_id)
            return fragment

        # Download fragment
        fragment = self._download_fragment(fragment_id)

        # Cache the fragment
        self._save_fragment_to_cache(fragment_id, fragment)

        # Start prefetching
        self._prefetch_adjacent_fragments(fragment_id)

        return fragment

    def get_manifest_info(self) -> Dict:
        """Get information about the manifest"""
        if not self.manifest:
            raise RuntimeError("Manifest not loaded")

        return {
            "total_vectors": self.manifest["metadata"]["total_vectors"],
            "vector_dim": self.manifest["metadata"]["vector_dim"],
            "chunk_size": self.manifest["metadata"]["chunk_size"],
            "num_fragments": len(self.manifest["metadata"]["fragments"]),
            "file_paths": {
                "mp4": self.mp4_path,
                "manifest": self.manifest_path,
                "faiss": self.faiss_path
            },
            "is_remote": {
                "mp4": self.is_mp4_remote,
                "manifest": self.is_manifest_remote,
                "faiss": self.is_faiss_remote
            }
        }

    def get_vectors_by_ids(self, vector_ids: List[int]) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve specific vectors by their IDs"""
        if not self.manifest:
            raise RuntimeError("Manifest not loaded")

        vectors_list = []
        metadata_list = []

        # Group by fragment for efficient access
        fragment_groups = {}
        for vec_id in vector_ids:
            if str(vec_id) not in self.manifest["vector_map"]:
                print(f"Warning: Vector ID {vec_id} not found")
                continue

            vec_info = self.manifest["vector_map"][str(vec_id)]
            frag_id = vec_info["fragment_id"]
            local_offset = vec_info["local_offset"]

            if frag_id not in fragment_groups:
                fragment_groups[frag_id] = []
            fragment_groups[frag_id].append((vec_id, local_offset))

        # Fetch vectors from each fragment
        for frag_id, vec_list in fragment_groups.items():
            fragment = self._get_fragment(frag_id)

            for vec_id, local_offset in vec_list:
                vectors_list.append(fragment["vectors"][local_offset])
                metadata_list.append(fragment["metadata"][local_offset])

        return np.array(vectors_list) if vectors_list else np.array([]), metadata_list

    def get_vectors_by_topic(self, topic: str) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve all vectors from a specific topic"""
        if not self.manifest:
            raise RuntimeError("Manifest not loaded")

        vectors_list = []
        metadata_list = []

        # Search through all fragments for matching topics
        for frag_info in self.manifest["metadata"]["fragments"]:
            if topic in frag_info.get("topics", []):
                fragment = self._get_fragment(frag_info["id"])
                # Filter by topic within fragment
                for i, meta in enumerate(fragment["metadata"]):
                    if meta.get("topic") == topic:
                        vectors_list.append(fragment["vectors"][i])
                        metadata_list.append(meta)

        return np.array(vectors_list) if vectors_list else np.array([]), metadata_list

    def search_vectors(self, query_vector: np.ndarray, top_k: int = 10, topic: str = None) -> List[Dict]:
        """Search using Faiss index with optional topic filtering"""
        if not self.faiss_index:
            raise RuntimeError("Faiss index not loaded. Cannot perform search.")

        # Normalize query vector for cosine similarity
        query_normalized = query_vector.copy().astype(np.float32)
        faiss.normalize_L2(query_normalized.reshape(1, -1))

        # Search in Faiss index
        similarities, indices = self.faiss_index.search(
            query_normalized.reshape(1, -1),
            top_k * 2  # Get more for filtering
        )

        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # Faiss uses -1 for invalid results
                continue

            # Get vector metadata
            vector_info = self.manifest["vector_map"].get(str(idx))
            if not vector_info:
                continue

            metadata = vector_info["metadata"]

            # Filter by topic if specified
            if topic and metadata.get("topic") != topic:
                continue

            # Get the actual vector for result
            fragment_id = vector_info["fragment_id"]
            local_offset = vector_info["local_offset"]
            fragment = self._get_fragment(fragment_id)
            vector = fragment["vectors"][local_offset]

            results.append({
                "vector": vector,
                "metadata": metadata,
                "similarity": float(sim),
                "vector_id": idx
            })

            if len(results) >= top_k:
                break

        return results

    def cleanup(self):
        """Clean up resources and background threads"""
        # Cancel any pending prefetch operations
        for future in self.prefetch_futures.values():
            future.cancel()

        # Shutdown executor
        self.executor.shutdown(wait=False)

        print("Decoder cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Convenience functions for common use cases
def create_cached_decoder(mp4_path: str,
                          cache_dir: Optional[str] = None,
                          **kwargs) -> CDNVectorMP4Decoder:
    """Create a cached decoder with reasonable defaults"""
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="vector_cache_")

    return CDNVectorMP4Decoder(
        mp4_path=mp4_path,
        disk_cache_dir=cache_dir,
        cache_size=100,
        enable_prefetching=True,
        **kwargs
    )


def create_cdn_decoder(base_url: str,
                       filename_prefix: str,
                       cache_dir: Optional[str] = None,
                       **kwargs) -> CDNVectorMP4Decoder:
    """Create a decoder specifically for CDN-hosted files"""
    mp4_url = f"{base_url}/{filename_prefix}.mp4"
    manifest_url = f"{base_url}/{filename_prefix}_manifest.json"
    faiss_url = f"{base_url}/{filename_prefix}_faiss.index"

    return create_cached_decoder(
        mp4_path=mp4_url,
        manifest_path=manifest_url,
        faiss_path=faiss_url,
        cache_dir=cache_dir,
        **kwargs
    )