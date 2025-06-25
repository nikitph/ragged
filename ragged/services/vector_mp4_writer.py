# In refactored_pipeline.py (or wherever VectorMP4Writer is defined)

import json
import logging
import struct
from dataclasses import asdict
from typing import Dict, List

import faiss
import numpy as np


# Assuming TextChunk is defined elsewhere
# from .data_models import TextChunk

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

    def _serialize_fragment_payload(self, vectors: np.ndarray, metadata: List[Dict]) -> bytes:
        """Serializes the data for one fragment (vectors + metadata) into a binary format."""
        vectors_data = vectors.astype(np.float32).tobytes()
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_size = len(metadata_json)

        # Pack data: vector bytes followed by metadata size and metadata bytes
        return vectors_data + struct.pack('>I', metadata_size) + metadata_json

    ### --- THIS IS THE MISSING METHOD --- ###
    def write_from_grouped_data(self, output_path: str, grouped_data: List[Dict], faiss_index: faiss.Index):
        """
        Writes the MP4, manifest, and index files from data that is pre-grouped by article.
        This is the primary method used by the hierarchical pipeline.

        Args:
            output_path: Path for the output .mp4 file.
            grouped_data: A list of dicts, where each dict is an article containing
                          {'id', 'vectors', 'metadata' (list of TextChunks)}.
            faiss_index: The pre-built Faiss index containing ALL vectors.
        """
        logging.info(f"Starting to write MP4 file to {output_path} from {len(grouped_data)} groups...")

        # Standard MP4 file headers
        ftyp_box = self._create_mp4_box('ftyp', b'isomiso2avc1mp41')
        moov_box = self._create_mp4_box('moov', b'')  # Empty moov box for simplicity

        # Prepare all fragments (one per article/group) before writing
        serialized_fragments = []
        total_vectors = 0
        for article_group in grouped_data:
            # The payload for the 'mdat' box
            fragment_payload = self._serialize_fragment_payload(
                article_group['vectors'],
                [asdict(m) for m in article_group['metadata']]
            )
            # The 'moof' (movie fragment) box contains the fragment's sequence number (ID)
            moof_box = self._create_mp4_box('moof', struct.pack('>I', article_group['id']))
            # The 'mdat' (media data) box contains the actual vectors and metadata
            mdat_box = self._create_mp4_box('mdat', fragment_payload)

            serialized_fragments.append({'moof': moof_box, 'mdat': mdat_box})
            total_vectors += len(article_group['vectors'])

        # Create the comprehensive manifest file
        manifest = {
            "metadata": {
                "vector_dim": self.vector_dim,
                "total_vectors": total_vectors,
                "total_fragments": len(grouped_data),
                "faiss_index_type": "IndexFlatIP" if not hasattr(faiss_index, 'nlist') else "IndexIVFPQ",
                "file_layout": {
                    "fragments": []
                }
            }
        }

        # Calculate byte offsets for each fragment in the manifest
        current_offset = len(ftyp_box) + len(moov_box)
        for i, frag_boxes in enumerate(serialized_fragments):
            article_group = grouped_data[i]
            moof_size = len(frag_boxes['moof'])
            mdat_size = len(frag_boxes['mdat'])

            manifest['metadata']['file_layout']['fragments'].append({
                "fragment_id": article_group['id'],
                "source": article_group['metadata'][0].source,  # Source is the same for all chunks
                "vector_count": len(article_group['vectors']),
                "moof_offset": current_offset,
                "mdat_offset": current_offset + moof_size,
            })
            current_offset += moof_size + mdat_size

        # --- Write all artifacts to disk ---

        # 1. Write the MP4 file
        with open(output_path, 'wb') as f:
            f.write(ftyp_box)
            f.write(moov_box)
            for frag_boxes in serialized_fragments:
                f.write(frag_boxes['moof'])
                f.write(frag_boxes['mdat'])
        logging.info(f"Successfully wrote MP4 file to {output_path}")

        # 2. Write the JSON manifest
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Manifest saved to {manifest_path}")

        # 3. Write the Faiss index
        faiss_path = output_path.replace('.mp4', '_faiss.index')
        faiss.write_index(faiss_index, faiss_path)
        logging.info(f"Faiss index saved to {faiss_path}")
    ### --- END OF MISSING METHOD --- ###