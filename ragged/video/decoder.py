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
from functools import lru_cache
from urllib.parse import urlparse
import threading
from concurrent.futures import ThreadPoolExecutor


class CachedVectorMP4Decoder:
    """Enhanced decoder with CDN optimization, caching, and HTTP range requests"""

    def __init__(self,
                 mp4_path: str,
                 manifest_path: Optional[str] = None,
                 faiss_path: Optional[str] = None,
                 cache_size: int = 100,
                 disk_cache_dir: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize enhanced decoder with caching capabilities

        Args:
            mp4_path: Path or URL to MP4 file
            manifest_path: Path or URL to manifest file (optional)
            faiss_path: Path or URL to Faiss index file (optional)
            cache_size: Maximum number of fragments to cache in memory
            disk_cache_dir: Directory for persistent disk cache (optional)
            max_retries: Maximum number of retry attempts for downloads
            timeout: Request timeout in seconds
        """
        self.mp4_path = mp4_path
        self.manifest_path = manifest_path or self._get_companion_path(mp4_path, '_manifest.json')
        self.faiss_path = faiss_path or self._get_companion_path(mp4_path, '_faiss.index')

        self.cache_size = cache_size
        self.disk_cache_dir = disk_cache_dir
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize caches
        self.memory_cache = {}
        self.manifest = None
        self.faiss_index = None

        # Check if we're dealing with URLs or local files
        self.is_remote = self._is_url(mp4_path)

        # Setup disk cache if requested
        if disk_cache_dir:
            try:
                os.makedirs(disk_cache_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not create disk cache directory {disk_cache_dir}: {e}")
                print("Continuing without disk cache...")
                self.disk_cache_dir = None

        # Load manifest and index
        self._load_manifest()
        self._load_faiss_index()

    def _is_url(self, path: str) -> bool:
        """Check if path is a URL"""
        parsed = urlparse(path)
        return parsed.scheme in ('http', 'https')

    def _get_companion_path(self, main_path: str, suffix: str) -> str:
        """Generate companion file path (manifest/faiss) from main MP4 path"""
        if self._is_url(main_path):
            return main_path.replace('.mp4', suffix)
        else:
            return main_path.replace('.mp4', suffix)

    def _get_cache_key(self, fragment_id: int) -> str:
        """Generate cache key for fragment"""
        path_hash = hashlib.md5(self.mp4_path.encode()).hexdigest()[:8]
        return f"frag_{fragment_id}_{path_hash}"

    def load_manifest_from_url(self, manifest_url: str):
        """Load manifest from separate CDN endpoint"""
        try:
            response = requests.get(manifest_url, timeout=self.timeout)
            response.raise_for_status()
            self.manifest = response.json()
            print(f"Loaded manifest from URL: {manifest_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to load manifest from {manifest_url}: {e}")

    def _load_manifest(self):
        """Load manifest from file or MP4"""
        if self._is_url(self.manifest_path):
            self.load_manifest_from_url(self.manifest_path)
        elif os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            # Extract from MP4 file
            self.manifest = self._extract_manifest_from_mp4()

    def _load_faiss_index(self):
        """Load Faiss index from file or URL"""
        try:
            if self._is_url(self.faiss_path):
                # Download Faiss index to temporary file
                temp_path = f"/tmp/faiss_index_{int(time.time())}.index"
                self._download_file(self.faiss_path, temp_path)
                self.faiss_index = faiss.read_index(temp_path)
                os.remove(temp_path)
                print(f"Downloaded and loaded Faiss index from {self.faiss_path}")
            elif os.path.exists(self.faiss_path):
                self.faiss_index = faiss.read_index(self.faiss_path)
                print(f"Loaded Faiss index from {self.faiss_path}")
        except Exception as e:
            print(f"Warning: Could not load Faiss index: {e}")
            self.faiss_index = None

    def _download_file(self, url: str, local_path: str):
        """Download complete file from URL"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def download_fragment_range(self, start_byte: int, end_byte: int) -> bytes:
        """Download specific byte range from CDN with retry logic"""
        for attempt in range(self.max_retries):
            try:
                headers = {'Range': f'bytes={start_byte}-{end_byte}'}
                response = requests.get(self.mp4_path, headers=headers, timeout=self.timeout)

                if response.status_code == 206:  # Partial Content
                    return response.content
                elif response.status_code == 416:  # Range Not Satisfiable
                    raise ValueError(f"Invalid byte range: {start_byte}-{end_byte}")
                elif response.status_code == 200:
                    # Server doesn't support range requests, get full content
                    print("Warning: Server doesn't support range requests")
                    return response.content[start_byte:end_byte + 1]
                else:
                    response.raise_for_status()

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to download range after {self.max_retries} attempts: {e}")
                wait_time = 2 ** attempt
                print(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)

    def _extract_manifest_from_mp4(self) -> Dict:
        """Extract manifest from MP4 file (local or remote)"""
        if self.is_remote:
            # For remote files, we need to find the manifest box
            # This is a simplified approach - in production you might want to
            # download the first few MB to locate the manifest
            raise NotImplementedError("Manifest extraction from remote MP4 not implemented. "
                                      "Please provide manifest_path parameter.")

        with open(self.mp4_path, 'rb') as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break

                box_size, box_type = struct.unpack('>I4s', header)
                box_type = box_type.decode('ascii')

                if box_type == 'manf':  # Manifest box
                    manifest_data = f.read(box_size - 8)
                    return json.loads(manifest_data.decode('utf-8'))
                else:
                    f.seek(box_size - 8, 1)  # Skip this box

        raise ValueError("No manifest found in MP4 file")

    @lru_cache(maxsize=100)
    def _read_fragment_cached(self, fragment_id: int) -> Dict:
        """Read fragment with multi-level caching"""
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
                        self.memory_cache[cache_key] = fragment
                        return fragment
                except Exception as e:
                    print(f"Failed to load from disk cache: {e}")

        # Download and cache
        fragment = self._download_fragment(fragment_id)

        # Store in memory cache (with size limit)
        if len(self.memory_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        self.memory_cache[cache_key] = fragment

        # Save to disk cache
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(fragment, f)
            except Exception as e:
                print(f"Failed to save to disk cache: {e}")

        return fragment

    def _download_fragment(self, fragment_id: int) -> Dict:
        """Download a specific fragment using byte ranges"""
        if not self.manifest:
            raise RuntimeError("Manifest not loaded")

        # Find fragment info in manifest
        fragment_info = None
        for frag in self.manifest["metadata"]["fragments"]:
            if frag["id"] == fragment_id:
                fragment_info = frag
                break

        if not fragment_info:
            raise ValueError(f"Fragment {fragment_id} not found in manifest")

        if self.is_remote:
            # Use HTTP range requests for remote files
            start_byte = fragment_info["byte_start"]
            end_byte = fragment_info["byte_end"] - 1  # HTTP ranges are inclusive
            fragment_data = self.download_fragment_range(start_byte, end_byte)

            # Extract mdat content (skip moof box)
            moof_size = fragment_info["mdat_start"] - fragment_info["byte_start"]
            mdat_content = fragment_data[moof_size + 8:]  # Skip mdat header

            return self._parse_fragment_data(mdat_content)
        else:
            # Local file access
            return self._read_fragment_local(fragment_id)

    def _read_fragment_local(self, fragment_id: int) -> Dict:
        """Read fragment from local file"""
        fragment_info = None
        for frag in self.manifest["metadata"]["fragments"]:
            if frag["id"] == fragment_id:
                fragment_info = frag
                break

        if not fragment_info:
            raise ValueError(f"Fragment {fragment_id} not found")

        with open(self.mp4_path, 'rb') as f:
            # Seek to mdat data start
            f.seek(fragment_info["data_start"])
            fragment_data = f.read(fragment_info["data_size"])

        return self._parse_fragment_data(fragment_data)

    def _parse_fragment_data(self, data: bytes) -> Dict:
        """Parse binary fragment data"""
        try:
            offset = 0

            # Validate we have enough data for header size
            if len(data) < 4:
                raise ValueError(f"Fragment data too short: {len(data)} bytes")

            # Read header
            header_size = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            # Validate header size is reasonable
            if header_size > len(data) - offset or header_size <= 0:
                raise ValueError(f"Invalid header size: {header_size}, remaining data: {len(data) - offset}")

            header_json = data[offset:offset + header_size].decode('utf-8')
            header = json.loads(header_json)
            offset += header_size

            # Read vectors
            if offset + 4 > len(data):
                raise ValueError("Not enough data for vectors size")

            vectors_size = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            if vectors_size > len(data) - offset or vectors_size <= 0:
                raise ValueError(f"Invalid vectors size: {vectors_size}, remaining data: {len(data) - offset}")

            vectors_data = data[offset:offset + vectors_size]
            vectors = np.frombuffer(vectors_data, dtype=np.float32).reshape(-1, header['vector_dim'])
            offset += vectors_size

            # Read metadata
            if offset + 4 > len(data):
                raise ValueError("Not enough data for metadata size")

            metadata_size = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            if metadata_size > len(data) - offset or metadata_size < 0:
                raise ValueError(f"Invalid metadata size: {metadata_size}, remaining data: {len(data) - offset}")

            metadata_json = data[offset:offset + metadata_size].decode('utf-8')
            metadata = json.loads(metadata_json)

            return {
                "header": header,
                "vectors": vectors,
                "metadata": metadata
            }

        except (UnicodeDecodeError, json.JSONDecodeError, struct.error) as e:
            raise ValueError(
                f"Failed to parse fragment data: {e}. Data length: {len(data)}, first 50 bytes: {data[:50].hex()}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing fragment data: {e}")

    def _read_fragment_local(self, fragment_id: int) -> Dict:
        """Read fragment from local file"""
        fragment_info = None
        for frag in self.manifest["metadata"]["fragments"]:
            if frag["id"] == fragment_id:
                fragment_info = frag
                break

        if not fragment_info:
            raise ValueError(f"Fragment {fragment_id} not found")

        try:
            with open(self.mp4_path, 'rb') as f:
                # Seek to mdat data start
                f.seek(fragment_info["data_start"])
                fragment_data = f.read(fragment_info["data_size"])

                if len(fragment_data) != fragment_info["data_size"]:
                    raise ValueError(f"Expected {fragment_info['data_size']} bytes, got {len(fragment_data)}")

            return self._parse_fragment_data(fragment_data)

        except Exception as e:
            # Log debug info
            print(f"Error reading fragment {fragment_id}: {e}")
            print(f"Fragment info: {fragment_info}")
            raise

    def get_vectors_by_ids(self, vector_ids: List[int]) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve specific vectors by their IDs"""
        vectors_list = []
        metadata_list = []

        # Group by fragment for efficient access
        fragment_groups = {}
        for vec_id in vector_ids:
            if str(vec_id) not in self.manifest["vector_map"]:
                continue

            vec_info = self.manifest["vector_map"][str(vec_id)]
            frag_id = vec_info["fragment_id"]
            local_offset = vec_info["local_offset"]

            if frag_id not in fragment_groups:
                fragment_groups[frag_id] = []
            fragment_groups[frag_id].append((vec_id, local_offset))

        # Fetch vectors from each fragment using cache
        for frag_id, vec_list in fragment_groups.items():
            fragment = self._read_fragment_cached(frag_id)

            for vec_id, local_offset in vec_list:
                vectors_list.append(fragment["vectors"][local_offset])
                metadata_list.append(fragment["metadata"][local_offset])

        return np.array(vectors_list), metadata_list

    def get_vectors_by_topic(self, topic: str) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve all vectors from a specific topic"""
        vectors_list = []
        metadata_list = []

        # Search through all fragments for matching topics
        for frag_info in self.manifest["metadata"]["fragments"]:
            if topic in frag_info.get("topics", []):
                fragment = self._read_fragment_cached(frag_info["id"])
                # Filter by topic within fragment
                for i, meta in enumerate(fragment["metadata"]):
                    if meta.get("topic") == topic:
                        vectors_list.append(fragment["vectors"][i])
                        metadata_list.append(meta)

        if vectors_list:
            return np.array(vectors_list), metadata_list
        else:
            return np.array([]), []

    def search_vectors(self, query_vector: np.ndarray, top_k: int = 10, topic: str = None) -> List[Dict]:
        """Search using Faiss index with optional topic filtering"""
        if not self.faiss_index:
            raise RuntimeError("Faiss index not loaded. Cannot perform search.")

        # Normalize query vector for cosine similarity
        query_normalized = query_vector.copy().astype(np.float32)
        faiss.normalize_L2(query_normalized.reshape(1, -1))

        # Search in Faiss index (get more results for potential filtering)
        search_k = top_k * 3 if topic else top_k
        similarities, indices = self.faiss_index.search(query_normalized.reshape(1, -1), search_k)

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

            # Get the actual vector for result (using cache)
            fragment_id = vector_info["fragment_id"]
            local_offset = vector_info["local_offset"]
            fragment = self._read_fragment_cached(fragment_id)
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

    def get_manifest_info(self) -> Dict:
        """Get information about the MP4 structure and contents"""
        if not self.manifest:
            return {}

        info = {
            "total_vectors": self.manifest["metadata"]["total_vectors"],
            "vector_dimension": self.manifest["metadata"]["vector_dim"],
            "total_fragments": len(self.manifest["metadata"]["fragments"]),
            "faiss_index_type": self.manifest["metadata"]["faiss_index_type"],
            "file_structure": self.manifest["metadata"]["file_structure"],
            "topics": set()
        }

        # Collect all topics
        for frag in self.manifest["metadata"]["fragments"]:
            info["topics"].update(frag.get("topics", []))

        info["topics"] = sorted(list(info["topics"]))

        return info

    def prefetch_fragments(self, fragment_ids: List[int]):
        """Prefetch multiple fragments in parallel"""

        def prefetch_single(frag_id):
            try:
                self._read_fragment_cached(frag_id)
            except Exception as e:
                print(f"Failed to prefetch fragment {frag_id}: {e}")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(prefetch_single, frag_id) for frag_id in fragment_ids]
            # Wait for all to complete
            for future in futures:
                future.result()

    def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self._read_fragment_cached.cache_clear()

        if self.disk_cache_dir and os.path.exists(self.disk_cache_dir):
            import shutil
            shutil.rmtree(self.disk_cache_dir)
            os.makedirs(self.disk_cache_dir, exist_ok=True)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        memory_size = len(self.memory_cache)
        lru_info = self._read_fragment_cached.cache_info()

        disk_files = 0
        if self.disk_cache_dir and os.path.exists(self.disk_cache_dir):
            disk_files = len([f for f in os.listdir(self.disk_cache_dir) if f.endswith('.pkl')])

        return {
            "memory_cache_size": memory_size,
            "memory_cache_limit": self.cache_size,
            "lru_cache_hits": lru_info.hits,
            "lru_cache_misses": lru_info.misses,
            "lru_cache_size": lru_info.currsize,
            "disk_cache_files": disk_files
        }


# Backward compatibility alias
VectorMP4Decoder = CachedVectorMP4Decoder


# Example usage
def demo_enhanced_decoder():
    """Demonstrate the enhanced decoder capabilities"""

    # Example 1: Local file with caching
    decoder_local = CachedVectorMP4Decoder(
        mp4_path="knowledge_base.mp4",
        cache_size=50,
        disk_cache_dir="./vector_cache"
    )

    print("=== Local Decoder Info ===")
    info = decoder_local.get_manifest_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Example 2: Remote file from CDN
    # decoder_remote = CachedVectorMP4Decoder(
    #     mp4_path="https://cdn.example.com/vectors/knowledge_base.mp4",
    #     manifest_path="https://cdn.example.com/vectors/knowledge_base_manifest.json",
    #     faiss_path="https://cdn.example.com/vectors/knowledge_base_faiss.index",
    #     cache_size=100,
    #     disk_cache_dir="./remote_cache"
    # )

    # Example 3: Search demonstration
    if decoder_local.faiss_index is not None:
        # Create a dummy query vector
        query = np.random.random(decoder_local.manifest["metadata"]["vector_dim"]).astype(np.float32)

        print("\n=== Search Results ===")
        results = decoder_local.search_vectors(query, top_k=5)

        for i, result in enumerate(results):
            print(f"Result {i + 1}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Topic: {result['metadata'].get('topic', 'Unknown')}")
            print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"  Text: {result['metadata'].get('text', '')[:100]}...")
            print()

    # Example 4: Cache statistics
    print("=== Cache Statistics ===")
    stats = decoder_local.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo_enhanced_decoder()