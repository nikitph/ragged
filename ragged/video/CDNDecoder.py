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
import networkx as nx
from dataclasses import dataclass


@dataclass
class KGSearchResult:
    """Result from knowledge graph search"""
    entities: List[Dict]
    relations: List[Dict]
    subgraph: nx.Graph
    fragment_id: int
    similarity_score: float = 0.0


class CDNVectorMP4Decoder:
    """Enhanced decoder with full CDN support, range requests, intelligent caching, and Knowledge Graph support"""

    def __init__(self,
                 mp4_path: str,
                 manifest_path: Optional[str] = None,
                 faiss_path: Optional[str] = None,
                 cache_size: int = 100,
                 disk_cache_dir: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 chunk_size: int = 8192,
                 enable_prefetching: bool = True,
                 enable_kg: bool = True):
        """
        Initialize enhanced decoder with full CDN capabilities and KG support

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
            enable_kg: Whether to enable knowledge graph functionality
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
        self.enable_kg = enable_kg

        # Initialize caches and state
        self.memory_cache = {}
        self.kg_memory_cache = {}  # Separate cache for KG fragments
        self.manifest = None
        self.faiss_index = None
        self.file_sizes = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prefetch_futures = {}
        self.kg_prefetch_futures = {}

        # Knowledge graph state
        self.global_graph = None
        self.entity_index = {}  # Entity text -> list of fragment IDs

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

        if self.enable_kg:
            self._initialize_kg_index()

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

            # Check for KG data
            if self.enable_kg and 'knowledge_graph' in self.manifest['metadata']:
                kg_meta = self.manifest['metadata']['knowledge_graph']
                if kg_meta.get('enabled', False):
                    print(
                        f"Knowledge graph enabled: {kg_meta['total_entities']} entities, {kg_meta['total_relations']} relations")
                else:
                    print("Knowledge graph not enabled in this file")
                    self.enable_kg = False
            else:
                print("No knowledge graph data found in manifest")
                self.enable_kg = False

        except Exception as e:
            raise RuntimeError(f"Failed to load manifest: {e}")

    def _load_faiss_index(self):
        """Load Faiss index from local file or CDN with persistent caching"""
        if not self.faiss_path:
            print("No Faiss index path provided")
            return

        cache_key = f"faiss_{hashlib.md5(self.faiss_path.encode()).hexdigest()}"

        # Try disk cache first
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.index")
            if os.path.exists(cache_file):
                try:
                    print(f"📄 Loading Faiss index from disk cache...")
                    self.faiss_index = faiss.read_index(cache_file)
                    print(f"✅ Loaded Faiss index from cache ({self.faiss_index.ntotal:,} vectors)")
                    return
                except Exception as e:
                    print(f"Warning: Failed to load Faiss index from cache: {e}")
                    # Remove corrupted cache file
                    try:
                        os.remove(cache_file)
                    except:
                        pass

        try:
            if self.is_faiss_remote:
                print(f"📥 Downloading Faiss index from {self.faiss_path}")
                index_size = self._get_file_size(self.faiss_path)
                print(f"   Index size: {index_size / 1024 / 1024:.1f} MB")

                index_data = self._stream_download(self.faiss_path, index_size)

                # Save to temporary file and load
                with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp_file:
                    tmp_file.write(index_data)
                    tmp_path = tmp_file.name

                self.faiss_index = faiss.read_index(tmp_path)
                os.unlink(tmp_path)
            else:
                if os.path.exists(self.faiss_path):
                    print(f"📄 Loading local Faiss index...")
                    self.faiss_index = faiss.read_index(self.faiss_path)
                else:
                    print(f"Faiss index not found at {self.faiss_path}")
                    return

            # Cache to disk for next time
            if self.disk_cache_dir and self.faiss_index:
                cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.index")
                try:
                    print(f"💾 Caching Faiss index to disk...")
                    faiss.write_index(self.faiss_index, cache_file)
                    print(f"✅ Faiss index cached to {cache_file}")
                except Exception as e:
                    print(f"Warning: Failed to cache Faiss index: {e}")

            print(f"✅ Faiss index ready: {self.faiss_index.ntotal:,} vectors")

        except Exception as e:
            print(f"Warning: Failed to load Faiss index: {e}")
            print("Search functionality will be limited without the index")

    def _initialize_kg_index(self):
        """Initialize knowledge graph index for fast entity lookups"""
        if not self.enable_kg or not self.manifest:
            return

        print("🔗 Initializing knowledge graph index...")

        kg_meta = self.manifest['metadata'].get('knowledge_graph', {})
        if not kg_meta.get('enabled', False):
            print("Knowledge graph not enabled in manifest")
            return

        # Build entity index for fast lookups
        for fragment_info in kg_meta.get('fragments', []):
            fragment_id = fragment_info['id']
            # We'll populate this when fragments are loaded
            # For now, just note that KG is available

        print(f"✅ Knowledge graph index ready")

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

    def _get_cache_key(self, fragment_id: int, is_kg: bool = False) -> str:
        """Generate cache key for fragment"""
        prefix = "kg_fragment" if is_kg else "fragment"
        return f"{prefix}_{fragment_id}_{hashlib.md5(self.mp4_path.encode()).hexdigest()[:8]}"

    def _get_fragment_from_cache(self, fragment_id: int, is_kg: bool = False) -> Optional[Dict]:
        """Try to get fragment from memory or disk cache"""
        cache_key = self._get_cache_key(fragment_id, is_kg)
        cache_dict = self.kg_memory_cache if is_kg else self.memory_cache

        # Check memory cache first
        if cache_key in cache_dict:
            return cache_dict[cache_key]

        # Check disk cache
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        fragment = pickle.load(f)
                        # Add to memory cache
                        self._add_to_memory_cache(cache_key, fragment, is_kg)
                        return fragment
                except:
                    pass

        return None

    def _add_to_memory_cache(self, cache_key: str, fragment: Dict, is_kg: bool = False):
        """Add fragment to memory cache with LRU eviction"""
        cache_dict = self.kg_memory_cache if is_kg else self.memory_cache

        # Simple FIFO eviction when cache is full
        while len(cache_dict) >= self.cache_size:
            oldest_key = next(iter(cache_dict))
            del cache_dict[oldest_key]

        cache_dict[cache_key] = fragment

    def _save_fragment_to_cache(self, fragment_id: int, fragment: Dict, is_kg: bool = False):
        """Save fragment to both memory and disk cache"""
        cache_key = self._get_cache_key(fragment_id, is_kg)

        # Add to memory cache
        self._add_to_memory_cache(cache_key, fragment, is_kg)

        # Save to disk cache
        if self.disk_cache_dir:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(fragment, f)
            except Exception as e:
                print(f"Failed to save fragment to disk cache: {e}")

    def _download_kg_fragment(self, fragment_id: int) -> Dict:
        """Download and parse a specific knowledge graph fragment"""
        if not self.manifest or not self.enable_kg:
            raise RuntimeError("Manifest not loaded or KG not enabled")

        # Find KG fragment info
        kg_fragments = self.manifest["metadata"]["knowledge_graph"].get("fragments", [])
        fragment_info = None
        for frag in kg_fragments:
            if frag["id"] == fragment_id:
                fragment_info = frag
                break

        if not fragment_info:
            raise ValueError(f"KG Fragment {fragment_id} not found in manifest")

        # Download KG fragment data using range request
        start_byte = fragment_info["byte_start"]
        end_byte = fragment_info["byte_end"] - 1  # HTTP ranges are inclusive

        print(f"Downloading KG fragment {fragment_id} (bytes {start_byte}-{end_byte})")
        fragment_data = self._download_range(self.mp4_path, start_byte, end_byte)

        # Parse the KG fragment
        return self._parse_kg_fragment_data(fragment_data, fragment_info)

    def _parse_kg_fragment_data(self, data: bytes, fragment_info: Dict) -> Dict:
        """Parse binary knowledge graph fragment data"""
        # Skip kgmf box to get to kgdt content
        kgdt_offset = fragment_info.get("mdat_start", 8) - fragment_info.get("byte_start", 0)
        kgdt_data = data[kgdt_offset + 8:]  # Skip kgdt header

        # Read header with size information
        if len(kgdt_data) < 12:
            raise ValueError("Invalid KG fragment data: too short")

        json_size, entity_count, relation_count = struct.unpack('<III', kgdt_data[:12])

        # Read JSON data
        json_data = kgdt_data[12:12 + json_size]
        fragment_data = json.loads(json_data.decode('utf-8'))

        return {
            "fragment_id": fragment_data["fragment_id"],
            "entities": fragment_data["entities"],
            "relations": fragment_data["relations"],
            "entity_count": fragment_data["entity_count"],
            "relation_count": fragment_data["relation_count"],
            "chunk_ids": set(fragment_data["chunk_ids"]),
            "timestamp": fragment_data["timestamp"]
        }

    def _download_fragment(self, fragment_id: int) -> Dict:
        """Download and parse a specific vector fragment"""
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

        print(f"Downloading vector fragment {fragment_id} (bytes {start_byte}-{end_byte})")
        fragment_data = self._download_range(self.mp4_path, start_byte, end_byte)

        # Parse the fragment
        return self._parse_fragment_data(fragment_data, fragment_info)

    def _parse_fragment_data(self, data: bytes, fragment_info: Dict, legacy: bool = False) -> Dict:
        """Parse binary vector fragment data"""
        if legacy:
            # For legacy format, data is already the mdat content
            mdat_data = data
        else:
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

    def _get_kg_fragment(self, fragment_id: int) -> Dict:
        """Get KG fragment with caching"""
        # Check cache first
        fragment = self._get_fragment_from_cache(fragment_id, is_kg=True)
        if fragment:
            return fragment

        # Download fragment
        fragment = self._download_kg_fragment(fragment_id)

        # Cache the fragment
        self._save_fragment_to_cache(fragment_id, fragment, is_kg=True)

        return fragment

    def _get_fragment(self, fragment_id: int) -> Dict:
        """Get vector fragment with caching and prefetching"""
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

    def _prefetch_adjacent_fragments(self, fragment_id: int):
        """Prefetch adjacent fragments in background"""
        if not self.enable_prefetching:
            return

        # Prefetch more aggressively for better search performance
        prefetch_range = 3  # Prefetch 3 fragments in each direction

        for offset in range(-prefetch_range, prefetch_range + 1):
            target_fragment_id = fragment_id + offset

            if target_fragment_id == fragment_id:
                continue  # Skip current fragment

            if target_fragment_id not in self.prefetch_futures:
                # Check if fragment exists
                if any(f["id"] == target_fragment_id for f in self.manifest["metadata"]["fragments"]):
                    future = self.executor.submit(self._get_fragment, target_fragment_id)
                    self.prefetch_futures[target_fragment_id] = future

    def get_manifest_info(self) -> Dict:
        """Get information about the manifest including KG data"""
        if not self.manifest:
            raise RuntimeError("Manifest not loaded")

        info = {
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

        # Add KG information if available
        if self.enable_kg and 'knowledge_graph' in self.manifest['metadata']:
            kg_meta = self.manifest['metadata']['knowledge_graph']
            info["knowledge_graph"] = {
                "enabled": kg_meta.get('enabled', False),
                "total_entities": kg_meta.get('total_entities', 0),
                "total_relations": kg_meta.get('total_relations', 0),
                "num_kg_fragments": len(kg_meta.get('fragments', [])),
                "entity_types": kg_meta.get('entity_types', {}),
                "relation_types": kg_meta.get('relation_types', {})
            }

        return info

    def get_vectors_by_ids(self, vector_ids: List[int]) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve specific vectors by their IDs with parallel downloads"""
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

        # Download fragments in parallel for better performance
        if len(fragment_groups) > 1:
            print(f"📥 Downloading {len(fragment_groups)} fragments in parallel...")

            # Submit all fragment downloads simultaneously
            future_to_fragment = {}
            for frag_id in fragment_groups.keys():
                future = self.executor.submit(self._get_fragment, frag_id)
                future_to_fragment[future] = frag_id

            # Collect results as they complete
            fragments = {}
            for future in as_completed(future_to_fragment.keys()):
                frag_id = future_to_fragment[future]
                try:
                    fragments[frag_id] = future.result()
                except Exception as e:
                    print(f"❌ Error downloading fragment {frag_id}: {e}")
                    continue
        else:
            # Single fragment - use normal method
            fragments = {}
            for frag_id in fragment_groups.keys():
                fragments[frag_id] = self._get_fragment(frag_id)

        # Extract vectors from fragments
        for frag_id, vec_list in fragment_groups.items():
            if frag_id in fragments:
                fragment = fragments[frag_id]
                for vec_id, local_offset in vec_list:
                    vectors_list.append(fragment["vectors"][local_offset])
                    metadata_list.append(fragment["metadata"][local_offset])

        return np.array(vectors_list) if vectors_list else np.array([]), metadata_list

    def search_entities(self, entity_text: str, entity_type: Optional[str] = None) -> List[KGSearchResult]:
        """Search for entities in the knowledge graph"""
        if not self.enable_kg:
            raise RuntimeError("Knowledge graph not enabled")

        results = []
        kg_fragments = self.manifest["metadata"]["knowledge_graph"].get("fragments", [])

        # Search through KG fragments
        for kg_frag_info in kg_fragments:
            kg_fragment = self._get_kg_fragment(kg_frag_info["id"])

            # Search entities in this fragment
            matching_entities = []
            for entity in kg_fragment["entities"]:
                if entity_text.lower() in entity["text"].lower():
                    if entity_type is None or entity["label"] == entity_type:
                        matching_entities.append(entity)

            if matching_entities:
                # Get related relations
                entity_ids = {e["entity_id"] for e in matching_entities}
                related_relations = []
                for relation in kg_fragment["relations"]:
                    if (relation["source_entity"] in entity_ids or
                            relation["target_entity"] in entity_ids):
                        related_relations.append(relation)

                # Build subgraph
                subgraph = nx.Graph()
                for entity in matching_entities:
                    subgraph.add_node(entity["entity_id"], **entity)

                for relation in related_relations:
                    if (subgraph.has_node(relation["source_entity"]) and
                            subgraph.has_node(relation["target_entity"])):
                        subgraph.add_edge(
                            relation["source_entity"],
                            relation["target_entity"],
                            **relation
                        )

                result = KGSearchResult(
                    entities=matching_entities,
                    relations=related_relations,
                    subgraph=subgraph,
                    fragment_id=kg_fragment["fragment_id"]
                )
                results.append(result)

        return results

    def search_relations(self, relation_type: str, source_entity: Optional[str] = None,
                         target_entity: Optional[str] = None) -> List[KGSearchResult]:
        """Search for relations in the knowledge graph"""
        if not self.enable_kg:
            raise RuntimeError("Knowledge graph not enabled")

        results = []
        kg_fragments = self.manifest["metadata"]["knowledge_graph"].get("fragments", [])

        # Search through KG fragments
        for kg_frag_info in kg_fragments:
            kg_fragment = self._get_kg_fragment(kg_frag_info["id"])

            # Search relations in this fragment
            matching_relations = []
            for relation in kg_fragment["relations"]:
                if relation["relation_type"] == relation_type:
                    # Check entity filters if provided
                    if source_entity:
                        # Find entity by text
                        source_matches = [e for e in kg_fragment["entities"]
                                          if e["entity_id"] == relation["source_entity"]
                                          and source_entity.lower() in e["text"].lower()]
                        if not source_matches:
                            continue

                    if target_entity:
                        # Find entity by text
                        target_matches = [e for e in kg_fragment["entities"]
                                          if e["entity_id"] == relation["target_entity"]
                                          and target_entity.lower() in e["text"].lower()]
                        if not target_matches:
                            continue

                    matching_relations.append(relation)

            if matching_relations:
                # Get related entities
                entity_ids = set()
                for relation in matching_relations:
                    entity_ids.add(relation["source_entity"])
                    entity_ids.add(relation["target_entity"])

                related_entities = [e for e in kg_fragment["entities"]
                                    if e["entity_id"] in entity_ids]

                # Build subgraph
                subgraph = nx.Graph()
                for entity in related_entities:
                    subgraph.add_node(entity["entity_id"], **entity)

                for relation in matching_relations:
                    if (subgraph.has_node(relation["source_entity"]) and
                            subgraph.has_node(relation["target_entity"])):
                        subgraph.add_edge(
                            relation["source_entity"],
                            relation["target_entity"],
                            **relation
                        )

                result = KGSearchResult(
                    entities=related_entities,
                    relations=matching_relations,
                    subgraph=subgraph,
                    fragment_id=kg_fragment["fragment_id"]
                )
                results.append(result)

        return results

    def get_entity_neighborhood(self, entity_text: str, depth: int = 1) -> Optional[KGSearchResult]:
        """Get the neighborhood of an entity up to a certain depth"""
        if not self.enable_kg:
            raise RuntimeError("Knowledge graph not enabled")

        # First find the entity
        entity_results = self.search_entities(entity_text)
        if not entity_results:
            return None

        # Combine all found entities and build expanded neighborhood
        all_entities = []
        all_relations = []
        combined_graph = nx.Graph()

        # Start with direct matches
        for result in entity_results:
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
            combined_graph = nx.union(combined_graph, result.subgraph)

        # Expand to specified depth
        current_entities = {e["entity_id"] for e in all_entities}

        for current_depth in range(depth):
            kg_fragments = self.manifest["metadata"]["knowledge_graph"].get("fragments", [])
            new_entities = set()

            for kg_frag_info in kg_fragments:
                kg_fragment = self._get_kg_fragment(kg_frag_info["id"])

                # Find relations connected to current entities
                for relation in kg_fragment["relations"]:
                    if relation["source_entity"] in current_entities:
                        new_entities.add(relation["target_entity"])
                        if relation not in all_relations:
                            all_relations.append(relation)
                    elif relation["target_entity"] in current_entities:
                        new_entities.add(relation["source_entity"])
                        if relation not in all_relations:
                            all_relations.append(relation)

                # Add new entities
                for entity in kg_fragment["entities"]:
                    if entity["entity_id"] in new_entities:
                        if entity not in all_entities:
                            all_entities.append(entity)
                            combined_graph.add_node(entity["entity_id"], **entity)

            # Add new relations to graph
            for relation in all_relations:
                if (combined_graph.has_node(relation["source_entity"]) and
                        combined_graph.has_node(relation["target_entity"])):
                    combined_graph.add_edge(
                        relation["source_entity"],
                        relation["target_entity"],
                        **relation
                    )

            current_entities.update(new_entities)

        return KGSearchResult(
            entities=all_entities,
            relations=all_relations,
            subgraph=combined_graph,
            fragment_id=-1  # Multiple fragments
        )

    def hybrid_search(self, query_vector: np.ndarray, entity_filter: Optional[str] = None,
                      relation_filter: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Perform hybrid search combining vector similarity and knowledge graph filtering"""
        if not self.faiss_index:
            raise RuntimeError("Faiss index not loaded. Cannot perform hybrid search.")

        # First, perform vector search
        vector_results = self.search_vectors(query_vector, top_k * 3)  # Get more for filtering

        if not self.enable_kg or (not entity_filter and not relation_filter):
            # No KG filtering requested, return vector results
            return vector_results[:top_k]

        # Filter results using knowledge graph
        filtered_results = []

        for result in vector_results:
            vector_metadata = result["metadata"]
            chunk_id = vector_metadata.get("chunk_id")

            if chunk_id is None:
                continue

            # Find KG fragments that contain this chunk
            relevant_kg_fragments = []
            kg_fragments = self.manifest["metadata"]["knowledge_graph"].get("fragments", [])

            for kg_frag_info in kg_fragments:
                kg_fragment = self._get_kg_fragment(kg_frag_info["id"])
                if chunk_id in kg_fragment["chunk_ids"]:
                    relevant_kg_fragments.append(kg_fragment)

            # Check if this result matches our KG filters
            matches_filter = False

            for kg_fragment in relevant_kg_fragments:
                # Check entity filter
                if entity_filter:
                    entity_match = any(
                        entity_filter.lower() in entity["text"].lower()
                        for entity in kg_fragment["entities"]
                        if entity["chunk_id"] == chunk_id
                    )
                    if not entity_match:
                        continue

                # Check relation filter
                if relation_filter:
                    relation_match = any(
                        relation["relation_type"] == relation_filter
                        for relation in kg_fragment["relations"]
                        if relation["chunk_id"] == chunk_id
                    )
                    if not relation_match:
                        continue

                matches_filter = True
                break

            if matches_filter or (not entity_filter and not relation_filter):
                # Add KG context to result
                kg_context = {
                    "entities": [],
                    "relations": []
                }

                for kg_fragment in relevant_kg_fragments:
                    chunk_entities = [e for e in kg_fragment["entities"] if e["chunk_id"] == chunk_id]
                    chunk_relations = [r for r in kg_fragment["relations"] if r["chunk_id"] == chunk_id]
                    kg_context["entities"].extend(chunk_entities)
                    kg_context["relations"].extend(chunk_relations)

                result["kg_context"] = kg_context
                filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        return filtered_results

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
        """Search using Faiss index with optional topic filtering and parallel fragment downloads"""
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

        # Collect all required fragments first
        required_fragments = set()
        vector_infos = []

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

            fragment_id = vector_info["fragment_id"]
            required_fragments.add(fragment_id)
            vector_infos.append((sim, idx, vector_info))

            if len(vector_infos) >= top_k:
                break

        # Download all required fragments in parallel
        fragments = {}
        if len(required_fragments) > 1:
            print(f"📥 Downloading {len(required_fragments)} fragments in parallel...")

            # Submit all fragment downloads simultaneously
            future_to_fragment = {}
            for frag_id in required_fragments:
                future = self.executor.submit(self._get_fragment, frag_id)
                future_to_fragment[future] = frag_id

            # Collect results as they complete
            for future in as_completed(future_to_fragment.keys()):
                frag_id = future_to_fragment[future]
                try:
                    fragments[frag_id] = future.result()
                except Exception as e:
                    print(f"❌ Error downloading fragment {frag_id}: {e}")
                    continue
        else:
            # Single fragment or all cached
            for frag_id in required_fragments:
                fragments[frag_id] = self._get_fragment(frag_id)

        # Build results using downloaded fragments
        results = []
        for sim, idx, vector_info in vector_infos:
            fragment_id = vector_info["fragment_id"]
            local_offset = vector_info["local_offset"]

            if fragment_id in fragments:
                fragment = fragments[fragment_id]
                vector = fragment["vectors"][local_offset]
                metadata = vector_info["metadata"]

                results.append({
                    "vector": vector,
                    "metadata": metadata,
                    "similarity": float(sim),
                    "vector_id": idx
                })

        return results

    def get_kg_statistics(self) -> Dict:
        """Get comprehensive knowledge graph statistics"""
        if not self.enable_kg:
            return {"enabled": False}

        kg_meta = self.manifest["metadata"]["knowledge_graph"]

        # Load all KG fragments to get detailed stats
        all_entities = []
        all_relations = []

        for kg_frag_info in kg_meta.get("fragments", []):
            kg_fragment = self._get_kg_fragment(kg_frag_info["id"])
            all_entities.extend(kg_fragment["entities"])
            all_relations.extend(kg_fragment["relations"])

        # Calculate statistics
        entity_types = {}
        relation_types = {}
        chunk_coverage = set()

        for entity in all_entities:
            entity_type = entity["label"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            chunk_coverage.add(entity["chunk_id"])

        for relation in all_relations:
            rel_type = relation["relation_type"]
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            chunk_coverage.add(relation["chunk_id"])

        return {
            "enabled": True,
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "unique_entity_types": len(entity_types),
            "unique_relation_types": len(relation_types),
            "entity_type_distribution": entity_types,
            "relation_type_distribution": relation_types,
            "chunks_with_kg_data": len(chunk_coverage),
            "kg_fragments": len(kg_meta.get("fragments", [])),
            "coverage_percentage": (len(chunk_coverage) / self.manifest["metadata"]["total_vectors"]) * 100
        }

    def cleanup(self):
        """Clean up resources and background threads"""
        # Cancel any pending prefetch operations
        for future in self.prefetch_futures.values():
            future.cancel()

        for future in self.kg_prefetch_futures.values():
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
                          enable_kg: bool = True,
                          **kwargs) -> CDNVectorMP4Decoder:
    """Create a cached decoder with reasonable defaults and KG support"""
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="vector_cache_")

    return CDNVectorMP4Decoder(
        mp4_path=mp4_path,
        disk_cache_dir=cache_dir,
        cache_size=100,
        enable_prefetching=True,
        enable_kg=enable_kg,
        **kwargs
    )


def create_cdn_decoder(base_url: str,
                       filename_prefix: str,
                       cache_dir: Optional[str] = None,
                       enable_kg: bool = True,
                       **kwargs) -> CDNVectorMP4Decoder:
    """Create a decoder specifically for CDN-hosted files with KG support"""
    mp4_url = f"{base_url}/{filename_prefix}.mp4"
    manifest_url = f"{base_url}/{filename_prefix}_manifest.json"
    faiss_url = f"{base_url}/{filename_prefix}_faiss.index"

    return create_cached_decoder(
        mp4_path=mp4_url,
        manifest_path=manifest_url,
        faiss_path=faiss_url,
        cache_dir=cache_dir,
        enable_kg=enable_kg,
        **kwargs
    )