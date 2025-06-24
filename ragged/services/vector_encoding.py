import json
import struct
import hashlib
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from ragged.video.mp4_structure import create_mp4_box, create_video_boxes
from ragged.models.text_chunk import TextChunk


class VectorMP4Encoder:
    """Encoder for converting vectors to MP4 format with FAISS indexing"""

    def __init__(self, vector_dim: int = 384, chunk_size: int = 1000):
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.fragments = []
        self.all_vectors = []
        self.manifest = self._initialize_manifest()
        self.faiss_index = None

    def _initialize_manifest(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "vector_dim": self.vector_dim,
                "chunk_size": self.chunk_size,
                "total_vectors": 0,
                "fragments": [],
                "faiss_index_type": "IndexFlatIP",
                "file_structure": {
                    "ftyp_start": 0, "ftyp_size": 0,
                    "video_track_start": 0, "video_track_size": 0,
                    "manifest_start": 0, "manifest_size": 0,
                    "fragments_start": 0, "total_file_size": 0
                }
            },
            "vector_map": {}
        }

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors with metadata to the encoder"""
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {vectors.shape[1]}")

        self.all_vectors.extend(vectors)
        num_vectors = len(vectors)

        for i in range(0, num_vectors, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_vectors)
            fragment_id = len(self.fragments)
            start_vector_id = self.manifest["metadata"]["total_vectors"] + i

            fragment = {
                "id": fragment_id,
                "vectors": vectors[i:end_idx],
                "metadata": metadata[i:end_idx],
                "vector_count": end_idx - i,
                "start_idx": start_vector_id,
                "end_idx": start_vector_id + (end_idx - i) - 1
            }
            self.fragments.append(fragment)

            # Update manifest
            topics = list({m.get("topic", "") for m in metadata[i:end_idx] if m.get("topic")})
            self.manifest["metadata"]["fragments"].append({
                "id": fragment_id,
                "vector_count": fragment["vector_count"],
                "start_idx": fragment["start_idx"],
                "end_idx": fragment["end_idx"],
                "topics": topics,
                "byte_start": 0, "byte_end": 0, "byte_size": 0,
                "moof_start": 0, "mdat_start": 0, "data_start": 0, "data_size": 0
            })

            for j in range(fragment["vector_count"]):
                vector_id = start_vector_id + j
                self.manifest["vector_map"][vector_id] = {
                    "fragment_id": fragment_id,
                    "local_offset": j,
                    "metadata": metadata[i + j]
                }

        self.manifest["metadata"]["total_vectors"] += num_vectors

    def _build_faiss_index(self) -> Optional[faiss.Index]:
        """Build appropriate FAISS index based on dataset size"""
        if not self.all_vectors:
            return None

        vectors_array = np.array(self.all_vectors).astype(np.float32)
        faiss.normalize_L2(vectors_array)
        n_vectors = len(vectors_array)

        if n_vectors < 100000:
            index = faiss.IndexFlatIP(self.vector_dim)
            self.manifest["metadata"]["faiss_index_type"] = "IndexFlatIP"
        else:
            nlist = min(int(4 * np.sqrt(n_vectors)), n_vectors // 10)

            # Find largest m (8-1) that divides vector_dim
            m_candidates = [m for m in range(8, 0, -1) if self.vector_dim % m == 0]
            m = m_candidates[0] if m_candidates else 8

            quantizer = faiss.IndexFlatIP(self.vector_dim)
            index = faiss.IndexIVFPQ(quantizer, self.vector_dim, nlist, m, 8)
            index.train(vectors_array)
            index.nprobe = min(nlist, 10)  # Number of clusters to search

            self.manifest["metadata"].update({
                "faiss_index_type": "IndexIVFPQ",
                "faiss_params": {
                    "nlist": nlist,
                    "nprobe": index.nprobe,
                    "m": m,
                    "bits": 8
                }
            })

        index.add(vectors_array)
        return index

    def _serialize_fragment(self, fragment: Dict) -> bytes:
        """Serialize fragment data into binary format"""
        header = json.dumps({
            "id": fragment["id"],
            "vector_count": fragment["vector_count"],
            "vector_dim": self.vector_dim,
            "start_idx": fragment["start_idx"],
            "end_idx": fragment["end_idx"]
        }).encode('utf-8')

        vectors_data = fragment["vectors"].astype(np.float32).tobytes()
        metadata_json = json.dumps(fragment["metadata"]).encode('utf-8')

        return (
                struct.pack('<I', len(header)) + header +
                struct.pack('<I', len(vectors_data)) + vectors_data +
                struct.pack('<I', len(metadata_json)) + metadata_json
        )

    def encode_to_mp4(self, output_path: str):
        """Complete encoding pipeline to MP4 format"""
        self.faiss_index = self._build_faiss_index()

        # Create MP4 structure components
        ftyp_box = create_mp4_box('ftyp', b'isom' + struct.pack('>I', 512) + b'isommp41avc1')
        video_boxes = create_video_boxes(self.manifest["metadata"]["total_vectors"])

        # Serialize all fragments
        serialized_fragments = []
        for fragment in self.fragments:
            fragment_data = self._serialize_fragment(fragment)
            moof_header = struct.pack('>II', fragment["id"], fragment["vector_count"])
            moof_box = create_mp4_box('moof', moof_header)
            mdat_box = create_mp4_box('mdat', fragment_data)
            serialized_fragments.append({
                'moof': moof_box,
                'mdat': mdat_box,
                'data': fragment_data
            })

        # Write to file with byte position tracking
        with open(output_path, 'wb') as f:
            current_pos = 0

            # 1. Write ftyp box
            f.write(ftyp_box)
            self.manifest["metadata"]["file_structure"]["ftyp_start"] = current_pos
            self.manifest["metadata"]["file_structure"]["ftyp_size"] = len(ftyp_box)
            current_pos += len(ftyp_box)

            # 2. Write video track
            video_start = current_pos
            for box in video_boxes:
                f.write(box)
                current_pos += len(box)
            self.manifest["metadata"]["file_structure"]["video_track_start"] = video_start
            self.manifest["metadata"]["file_structure"]["video_track_size"] = current_pos - video_start

            # 3. Write manifest placeholder
            manifest_placeholder = create_mp4_box('manf', b'MANIFEST_PLACEHOLDER')
            f.write(manifest_placeholder)
            manifest_start = current_pos
            current_pos += len(manifest_placeholder)

            # 4. Write fragments
            fragments_start = current_pos
            for i, frag in enumerate(serialized_fragments):
                frag_start = current_pos
                f.write(frag['moof'])
                f.write(frag['mdat'])
                current_pos += len(frag['moof']) + len(frag['mdat'])

                # Update fragment positions
                self.manifest["metadata"]["fragments"][i].update({
                    "byte_start": frag_start,
                    "byte_end": current_pos,
                    "byte_size": current_pos - frag_start,
                    "moof_start": frag_start,
                    "mdat_start": frag_start + len(frag['moof']),
                    "data_start": frag_start + len(frag['moof']) + 8,
                    "data_size": len(frag['data'])
                })

            self.manifest["metadata"]["file_structure"]["fragments_start"] = fragments_start
            self.manifest["metadata"]["file_structure"]["total_file_size"] = current_pos

            # 5. Overwrite manifest with final version
            final_manifest = json.dumps(self.manifest, indent=2).encode('utf-8')
            manifest_box = create_mp4_box('manf', final_manifest)
            f.seek(manifest_start)
            f.write(manifest_box)

            # Pad if needed
            if len(manifest_box) < len(manifest_placeholder):
                f.write(b'\x00' * (len(manifest_placeholder) - len(manifest_box)))

        # Save artifacts
        self._save_artifacts(output_path)

    def _save_artifacts(self, output_path: str):
        """Save additional files (manifest and FAISS index)"""
        # Save manifest
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        # Save FAISS index if available
        if self.faiss_index:
            index_path = output_path.replace('.mp4', '_faiss.index')
            faiss.write_index(self.faiss_index, index_path)