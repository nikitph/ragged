import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib
import os
import faiss


class VectorMP4Decoder:
    """Decodes vectors from MP4 custom tracks"""

    def __init__(self, mp4_path: str, manifest_path: str = None, faiss_path: str = None):
        self.mp4_path = mp4_path
        self.manifest_path = manifest_path or mp4_path.replace('.mp4', '_manifest.json')
        self.faiss_path = faiss_path or mp4_path.replace('.mp4', '_faiss.index')
        self.manifest = None
        self.faiss_index = None
        self.fragments_cache = {}

        self._load_manifest()
        self._load_faiss_index()

    def _load_faiss_index(self):
        """Load Faiss index from separate file"""
        if os.path.exists(self.faiss_path):
            try:
                self.faiss_index = faiss.read_index(self.faiss_path)
                print(f"Loaded Faiss index from {self.faiss_path}")
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load Faiss index from {self.faiss_path}: {e}")
        else:
            raise FileNotFoundError(f"Faiss index not found at {self.faiss_path}. "
                                    f"Make sure to encode with VectorMP4Encoder first.")

    def _load_manifest(self):
        """Load manifest from file or MP4"""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            # Extract from MP4 file
            self.manifest = self._extract_manifest_from_mp4()

    def _extract_manifest_from_mp4(self) -> Dict:
        """Extract manifest from MP4 file"""
        with open(self.mp4_path, 'rb') as f:
            while True:
                # Read box header
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

    def _read_fragment(self, fragment_id: int) -> Dict:
        """Read and parse a specific fragment"""
        if fragment_id in self.fragments_cache:
            return self.fragments_cache[fragment_id]

        with open(self.mp4_path, 'rb') as f:
            fragment_found = False
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break

                box_size, box_type = struct.unpack('>I4s', header)
                box_type = box_type.decode('ascii')

                if box_type == 'moof':
                    moof_data = f.read(box_size - 8)
                    frag_id, vector_count = struct.unpack('>II', moof_data[:8])

                    if frag_id == fragment_id:
                        fragment_found = True
                        continue
                elif box_type == 'mdat' and fragment_found:
                    fragment_data = f.read(box_size - 8)
                    parsed_fragment = self._parse_fragment_data(fragment_data)
                    self.fragments_cache[fragment_id] = parsed_fragment
                    return parsed_fragment
                else:
                    f.seek(box_size - 8, 1)

        raise ValueError(f"Fragment {fragment_id} not found")

    def _parse_fragment_data(self, data: bytes) -> Dict:
        """Parse binary fragment data"""
        offset = 0

        # Read header
        header_size = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        header_json = data[offset:offset + header_size].decode('utf-8')
        header = json.loads(header_json)
        offset += header_size

        # Read vectors
        vectors_size = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        vectors_data = data[offset:offset + vectors_size]
        vectors = np.frombuffer(vectors_data, dtype=np.float32).reshape(-1, header['vector_dim'])
        offset += vectors_size

        # Read metadata
        metadata_size = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        metadata_json = data[offset:offset + metadata_size].decode('utf-8')
        metadata = json.loads(metadata_json)

        return {
            "header": header,
            "vectors": vectors,
            "metadata": metadata
        }

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

        # Fetch vectors from each fragment
        for frag_id, vec_list in fragment_groups.items():
            fragment = self._read_fragment(frag_id)

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
                fragment = self._read_fragment(frag_info["id"])
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
        """Search using Faiss index"""
        if not self.faiss_index:
            raise RuntimeError("Faiss index not loaded. Cannot perform search.")

        # Normalize query vector for cosine similarity
        query_normalized = query_vector.copy().astype(np.float32)
        faiss.normalize_L2(query_normalized.reshape(1, -1))

        # Search in Faiss index
        similarities, indices = self.faiss_index.search(query_normalized.reshape(1, -1),
                                                        top_k * 2)  # Get more for filtering

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
            fragment = self._read_fragment(fragment_id)
            vector = fragment["vectors"][local_offset]

            results.append({
                "vector": vector,
                "metadata": metadata,
                "similarity": float(sim)
            })

            if len(results) >= top_k:
                break

        return results