import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib
import os


class VectorMP4Encoder:
    """Encodes vectors into MP4 custom tracks for CDN distribution"""

    def __init__(self, vector_dim: int = 1536, chunk_size: int = 1000):
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.fragments = []
        self.manifest = {
            "metadata": {
                "vector_dim": vector_dim,
                "chunk_size": chunk_size,
                "total_vectors": 0,
                "fragments": []
            },
            "vector_map": {}  # vector_id -> fragment_info
        }

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors with metadata to be encoded"""
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {vectors.shape[1]}")

        # Create chunks in sequential order
        num_vectors = len(vectors)
        for i in range(0, num_vectors, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_vectors)
            chunk_vectors = vectors[i:end_idx]
            chunk_metadata = metadata[i:end_idx]

            fragment_id = len(self.fragments)
            start_vector_id = self.manifest["metadata"]["total_vectors"] + i
            end_vector_id = self.manifest["metadata"]["total_vectors"] + end_idx - 1

            fragment = {
                "id": fragment_id,
                "vectors": chunk_vectors,
                "metadata": chunk_metadata,
                "vector_count": len(chunk_vectors),
                "start_idx": start_vector_id,
                "end_idx": end_vector_id
            }

            self.fragments.append(fragment)

            # Update manifest
            frag_info = {
                "id": fragment_id,
                "vector_count": len(chunk_vectors),
                "start_idx": start_vector_id,
                "end_idx": end_vector_id,
                "topics": list(set([m.get("topic", "") for m in chunk_metadata if m.get("topic")]))
            }
            self.manifest["metadata"]["fragments"].append(frag_info)

            # Map individual vectors
            for j, vec_meta in enumerate(chunk_metadata):
                vector_id = start_vector_id + j
                self.manifest["vector_map"][vector_id] = {
                    "fragment_id": fragment_id,
                    "local_offset": j,
                    "metadata": vec_meta
                }

        self.manifest["metadata"]["total_vectors"] += num_vectors

    def _serialize_fragment(self, fragment: Dict) -> bytes:
        """Serialize a fragment into binary format"""
        # Fragment header
        header = {
            "id": fragment["id"],
            "vector_count": fragment["vector_count"],
            "vector_dim": self.vector_dim,
            "start_idx": fragment["start_idx"],
            "end_idx": fragment["end_idx"]
        }

        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json)

        # Serialize vectors (float32)
        vectors_data = fragment["vectors"].astype(np.float32).tobytes()

        # Serialize metadata
        metadata_json = json.dumps(fragment["metadata"]).encode('utf-8')
        metadata_size = len(metadata_json)

        # Pack: header_size(4) + header + vectors_size(4) + vectors + metadata_size(4) + metadata
        packed_data = (
                struct.pack('<I', header_size) + header_json +
                struct.pack('<I', len(vectors_data)) + vectors_data +
                struct.pack('<I', metadata_size) + metadata_json
        )

        return packed_data

    def _create_mp4_box(self, box_type: str, data: bytes) -> bytes:
        """Create MP4 box with type and data"""
        size = len(data) + 8  # 4 bytes size + 4 bytes type + data
        return struct.pack('>I', size) + box_type.encode('ascii') + data

    def encode_to_mp4(self, output_path: str):
        """Encode vectors to MP4 file with custom tracks"""

        # Create ftyp box (file type)
        ftyp_data = b'isom' + struct.pack('>I', 512) + b'isom'
        ftyp_box = self._create_mp4_box('ftyp', ftyp_data)

        # Create manifest as initialization segment
        manifest_data = json.dumps(self.manifest, indent=2).encode('utf-8')
        manifest_box = self._create_mp4_box('manf', manifest_data)  # Custom manifest box

        # Create fragments
        fragment_boxes = []
        for fragment in self.fragments:
            fragment_data = self._serialize_fragment(fragment)

            # Create moof box (movie fragment)
            moof_header = struct.pack('>II', fragment["id"], fragment["vector_count"])
            moof_box = self._create_mp4_box('moof', moof_header)

            # Create mdat box (media data)
            mdat_box = self._create_mp4_box('mdat', fragment_data)

            fragment_boxes.extend([moof_box, mdat_box])

        # Write MP4 file
        with open(output_path, 'wb') as f:
            f.write(ftyp_box)
            f.write(manifest_box)
            for box in fragment_boxes:
                f.write(box)

        # Write manifest separately for CDN optimization
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        print(f"Encoded {self.manifest['metadata']['total_vectors']} vectors to {output_path}")
        print(f"Created {len(self.fragments)} fragments")
        print(f"Manifest saved to {manifest_path}")


