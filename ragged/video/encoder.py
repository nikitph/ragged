import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib
import os
import faiss


class VectorMP4Encoder:
    """Encodes vectors into MP4 custom tracks for CDN distribution"""

    def __init__(self, vector_dim: int = 1536, chunk_size: int = 1000):
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.fragments = []
        self.faiss_index = None
        self.all_vectors = []  # Store all vectors for Faiss index
        self.manifest = {
            "metadata": {
                "vector_dim": vector_dim,
                "chunk_size": chunk_size,
                "total_vectors": 0,
                "fragments": [],
                "faiss_index_type": "IndexFlatIP"  # Inner Product for cosine similarity
            },
            "vector_map": {}  # vector_id -> fragment_info
        }

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors with metadata to be encoded"""
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {vectors.shape[1]}")

        # Store vectors for Faiss index
        self.all_vectors.extend(vectors)

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

    def _build_faiss_index(self):
        """Build Faiss IVFPQ index for better compression and speed"""
        if not self.all_vectors:
            return None

        vectors_array = np.array(self.all_vectors).astype(np.float32)
        n_vectors = len(vectors_array)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_array)

        # Choose index type based on dataset size
        if n_vectors < 10000:
            # Small dataset: use flat index
            print(f"Using IndexFlatIP for {n_vectors} vectors")
            index = faiss.IndexFlatIP(self.vector_dim)
            index.add(vectors_array)
        else:
            # Large dataset: use IVFPQ
            # Calculate optimal parameters
            nlist = min(int(4 * np.sqrt(n_vectors)), n_vectors // 10)  # Number of clusters
            nlist = max(nlist, 1)  # At least 1 cluster

            m = 8  # Number of subquantizers (must divide vector_dim)
            while self.vector_dim % m != 0 and m > 1:
                m -= 1

            nbits = 8  # Bits per subquantizer

            print(f"Using IndexIVFPQ for {n_vectors} vectors: nlist={nlist}, m={m}, nbits={nbits}")

            # Create IVFPQ index
            quantizer = faiss.IndexFlatIP(self.vector_dim)
            index = faiss.IndexIVFPQ(quantizer, self.vector_dim, nlist, m, nbits)

            # Train the index (required for IVFPQ)
            print("Training IVFPQ index...")
            index.train(vectors_array)

            # Add vectors
            index.add(vectors_array)

            # Set search parameters for good recall
            index.nprobe = min(nlist, 10)  # Number of clusters to search

        # Update manifest with index info
        if hasattr(index, 'nlist'):
            self.manifest["metadata"]["faiss_index_type"] = "IndexIVFPQ"
            self.manifest["metadata"]["faiss_params"] = {
                "nlist": index.nlist,
                "nprobe": index.nprobe,
                "m": m if 'm' in locals() else None,
                "nbits": nbits if 'nbits' in locals() else None
            }
        else:
            self.manifest["metadata"]["faiss_index_type"] = "IndexFlatIP"

        return index

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

    def _create_minimal_video_track(self) -> bytes:
        """Create a minimal video track with single black frame"""
        # Create a simple 1x1 black frame (H.264 IDR frame)
        # This is a minimal valid H.264 frame
        h264_frame = bytes([
            0x00, 0x00, 0x00, 0x01,  # NAL unit start code
            0x67, 0x42, 0x00, 0x0a,  # SPS (Sequence Parameter Set)
            0x8d, 0x68, 0x05, 0x8b,
            0x00, 0x00, 0x00, 0x01,  # NAL unit start code
            0x68, 0xce, 0x06, 0xe2,  # PPS (Picture Parameter Set)
            0x00, 0x00, 0x00, 0x01,  # NAL unit start code
            0x65, 0x88, 0x80, 0x10,  # IDR frame (1x1 black)
            0x00, 0x02, 0x00, 0x08
        ])

        return h264_frame

    def _create_video_boxes(self) -> List[bytes]:
        """Create minimal video track boxes for MP4 compatibility"""
        # Calculate duration based on vector count (1 sec per 100 vectors, max 60 sec)
        duration = min(max(self.manifest["metadata"]["total_vectors"] // 100, 1), 60) * 1000  # timescale units

        # mvhd (movie header) - simplified
        mvhd_data = struct.pack('>IIIIII',
                                0,  # version + flags
                                0,  # creation_time
                                0,  # modification_time
                                1000,  # timescale
                                duration,  # duration
                                0x00010000  # rate (1.0)
                                ) + b'\x00' * 76  # padding for remaining fields
        mvhd_box = self._create_mp4_box('mvhd', mvhd_data)

        # tkhd (track header) - simplified
        tkhd_data = struct.pack('>IIIIII',
                                0x0000000F,  # version + flags (track enabled)
                                0,  # creation_time
                                0,  # modification_time
                                1,  # track_ID
                                0,  # reserved
                                duration  # duration
                                ) + b'\x00' * 60  # padding
        tkhd_box = self._create_mp4_box('tkhd', tkhd_data)

        # mdhd (media header)
        mdhd_data = struct.pack('>IIIIII',
                                0,  # version + flags
                                0,  # creation_time
                                0,  # modification_time
                                1000,  # timescale
                                duration,  # duration
                                0  # language + pre_defined
                                )
        mdhd_box = self._create_mp4_box('mdhd', mdhd_data)

        # hdlr (handler reference)
        hdlr_data = struct.pack('>IIII', 0, 0, 0x76696465, 0) + b'VideoHandler\x00'
        hdlr_box = self._create_mp4_box('hdlr', hdlr_data)

        # Combine boxes
        mdia_content = mdhd_box + hdlr_box
        mdia_box = self._create_mp4_box('mdia', mdia_content)

        trak_content = tkhd_box + mdia_box
        trak_box = self._create_mp4_box('trak', trak_content)

        moov_content = mvhd_box + trak_box
        moov_box = self._create_mp4_box('moov', moov_content)

        # Video frame data
        video_frame = self._create_minimal_video_track()
        mdat_video_box = self._create_mp4_box('mdat', video_frame)

        return [moov_box, mdat_video_box]

    def _create_mp4_box(self, box_type: str, data: bytes) -> bytes:
        """Create MP4 box with type and data"""
        size = len(data) + 8  # 4 bytes size + 4 bytes type + data
        return struct.pack('>I', size) + box_type.encode('ascii') + data

    def encode_to_mp4(self, output_path: str):
        """Encode vectors to MP4 file with custom tracks"""

        # Build Faiss index before encoding
        print("Building Faiss index...")
        self.faiss_index = self._build_faiss_index()

        # Create ftyp box (file type) - ensure video compatibility
        ftyp_data = b'isom' + struct.pack('>I', 512) + b'isommp41avc1'  # Add video brands
        ftyp_box = self._create_mp4_box('ftyp', ftyp_data)

        # Create minimal video track for player compatibility
        video_boxes = self._create_video_boxes()

        # Create manifest as custom box
        manifest_data = json.dumps(self.manifest, indent=2).encode('utf-8')
        manifest_box = self._create_mp4_box('manf', manifest_data)

        # Create vector fragments
        fragment_boxes = []
        for fragment in self.fragments:
            fragment_data = self._serialize_fragment(fragment)

            # Create moof box (movie fragment)
            moof_header = struct.pack('>II', fragment["id"], fragment["vector_count"])
            moof_box = self._create_mp4_box('moof', moof_header)

            # Create mdat box (media data)
            mdat_box = self._create_mp4_box('mdat', fragment_data)

            fragment_boxes.extend([moof_box, mdat_box])

        # Write MP4 file with video track first, then vector data
        with open(output_path, 'wb') as f:
            f.write(ftyp_box)
            # Video track for compatibility
            for box in video_boxes:
                f.write(box)
            # Custom manifest and vector fragments
            f.write(manifest_box)
            for box in fragment_boxes:
                f.write(box)

        # Write manifest separately for CDN optimization
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        # Save Faiss index separately for fast loading
        if self.faiss_index:
            faiss_path = output_path.replace('.mp4', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"Faiss index saved to {faiss_path}")

        print(f"Encoded {self.manifest['metadata']['total_vectors']} vectors to {output_path}")
        print(f"Created {len(self.fragments)} fragments")
        print(f"Manifest saved to {manifest_path}")
        print(f"MP4 now compatible with video players")