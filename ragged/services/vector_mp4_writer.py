import json
import logging
import struct
from dataclasses import asdict
from typing import List, Dict

import faiss
import numpy as np

from ragged.models.text_chunk import TextChunk


class VectorMP4Writer:
    """
    Writes ordered vectors, metadata, and cluster info into an MP4 container,
    along with a JSON manifest and a Faiss index file.
    """

    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim

    def _create_mp4_box(self, box_type: str, data: bytes) -> bytes:
        """Creates a standard MP4 box."""
        size = len(data) + 8
        return struct.pack('>I', size) + box_type.encode('ascii') + data

    def _serialize_cluster_as_fragment(self, vectors: np.ndarray, metadata: List[Dict]) -> bytes:
        """Serializes the data for one cluster (fragment) into a binary format."""
        vectors_data = vectors.astype(np.float32).tobytes()
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_size = len(metadata_json)

        # Pack data: vector bytes followed by metadata size and metadata bytes
        return vectors_data + struct.pack('>I', metadata_size) + metadata_json

    def write(self, output_path: str, vectors: np.ndarray, metadata: List[TextChunk],
              cluster_info: List[Dict], faiss_index: faiss.Index):
        """
        Writes the MP4, manifest, and index files.

        Args:
            output_path: Path for the output .mp4 file.
            vectors: The final, ordered vector array.
            metadata: The final, ordered list of TextChunks.
            cluster_info: Information about each cluster (start index, count).
            faiss_index: The trained Faiss index.
        """
        logging.info(f"Starting to write MP4 file to {output_path}")

        # Basic MP4 header
        ftyp_data = b'isom' + struct.pack('>I', 512) + b'isomiso2avc1mp41'
        ftyp_box = self._create_mp4_box('ftyp', ftyp_data)

        # Minimal moov box for compatibility
        moov_box = self._create_mp4_box('moov', b'')  # Empty for simplicity

        # Prepare fragments (one per cluster)
        serialized_fragments = []
        for cluster in cluster_info:
            start = cluster['start_index']
            end = start + cluster['vector_count']

            cluster_vectors = vectors[start:end]
            cluster_metadata = [asdict(m) for m in metadata[start:end]]

            fragment_data = self._serialize_cluster_as_fragment(cluster_vectors, cluster_metadata)

            # moof (movie fragment) box contains sequence number
            moof_box = self._create_mp4_box('moof', struct.pack('>I', cluster['cluster_id']))
            # mdat (media data) box contains the payload
            mdat_box = self._create_mp4_box('mdat', fragment_data)

            serialized_fragments.append({'moof': moof_box, 'mdat': mdat_box})

        # Create the manifest
        manifest = {
            "metadata": {
                "vector_dim": self.vector_dim,
                "total_vectors": len(vectors),
                "total_clusters": len(cluster_info),
                "faiss_index_type": "IndexIVFPQ",
                "file_layout": {
                    "ftyp": {"offset": 0, "size": len(ftyp_box)},
                    "moov": {"offset": len(ftyp_box), "size": len(moov_box)},
                    "fragments": []
                }
            }
        }

        # Calculate byte offsets for the manifest
        current_offset = len(ftyp_box) + len(moov_box)
        for i, frag in enumerate(serialized_fragments):
            moof_size = len(frag['moof'])
            mdat_size = len(frag['mdat'])

            manifest['metadata']['file_layout']['fragments'].append({
                "cluster_id": cluster_info[i]['cluster_id'],
                "vector_count": cluster_info[i]['vector_count'],
                "moof_offset": current_offset,
                "mdat_offset": current_offset + moof_size,
                "fragment_size": moof_size + mdat_size
            })
            current_offset += moof_size + mdat_size

        # Create manifest box ('meta' with a 'manf' handler)
        manf_json = json.dumps(manifest, indent=2).encode('utf-8')
        meta_box = self._create_mp4_box('meta', self._create_mp4_box('manf', manf_json))

        # Write the final MP4 file
        with open(output_path, 'wb') as f:
            f.write(ftyp_box)
            f.write(moov_box)
            # The manifest could also be here if needed, but a separate file is often better
            for frag in serialized_fragments:
                f.write(frag['moof'])
                f.write(frag['mdat'])

        logging.info(f"Successfully wrote {len(vectors)} vectors in {len(cluster_info)} fragments to {output_path}")

        # Save separate manifest file
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Manifest saved to {manifest_path}")

        # Save Faiss index
        faiss_path = output_path.replace('.mp4', '_faiss.index')
        faiss.write_index(faiss_index, faiss_path)
        logging.info(f"Faiss index saved to {faiss_path}")