import json
import struct
import os
import faiss
import numpy as np
from typing import Dict, List, Any, Optional
from .mp4_structure import create_mp4_box, create_video_boxes


class VectorMP4Encoder:
    """Handles final MP4 file assembly and writing with byte position tracking"""

    def assemble_mp4(self,
                     manifest: Dict[str, Any],
                     fragments: List[Dict],
                     faiss_index: Optional[faiss.Index],
                     output_path: str) -> None:
        """
        Assemble and write the complete MP4 file with precise byte tracking

        Args:
            manifest: The manifest dictionary to be updated with byte positions
            fragments: List of fragment dictionaries containing serialized data
            faiss_index: Faiss index for vector search (optional)
            output_path: Path to the output MP4 file
        """
        # Create MP4 boxes
        ftyp_box = create_mp4_box('ftyp', b'isom' + struct.pack('>I', 512) + b'isommp41avc1')
        video_boxes = create_video_boxes(manifest["metadata"]["total_vectors"])

        # Open output file
        with open(output_path, 'wb') as f:
            current_position = 0

            # 1. Write ftyp box
            f.write(ftyp_box)
            manifest['metadata']['file_structure']['ftyp_start'] = current_position
            manifest['metadata']['file_structure']['ftyp_size'] = len(ftyp_box)
            current_position += len(ftyp_box)

            # 2. Write video track boxes
            video_start = current_position
            for box in video_boxes:
                f.write(box)
                current_position += len(box)
            video_size = current_position - video_start
            manifest['metadata']['file_structure']['video_track_start'] = video_start
            manifest['metadata']['file_structure']['video_track_size'] = video_size

            # 3. Write manifest placeholder (will be overwritten later)
            manifest_placeholder = create_mp4_box('manf', b'MANIFEST_PLACEHOLDER')
            f.write(manifest_placeholder)
            manifest_start = current_position
            current_position += len(manifest_placeholder)

            # 4. Write fragments and update manifest with byte positions
            fragments_start = current_position
            for i, frag in enumerate(fragments):
                fragment_start = current_position

                # Write moof box
                f.write(frag['moof'])
                current_position += len(frag['moof'])
                mdat_start = current_position

                # Write mdat box
                f.write(frag['mdat'])
                current_position += len(frag['mdat'])

                # Update fragment info in manifest
                frag_info = manifest['metadata']['fragments'][i]
                frag_info.update({
                    "byte_start": fragment_start,
                    "byte_end": current_position,
                    "byte_size": current_position - fragment_start,
                    "moof_start": fragment_start,
                    "mdat_start": mdat_start,
                    "data_start": mdat_start + 8,  # Skip mdat header
                    "data_size": len(frag['data'])
                })

            fragments_size = current_position - fragments_start
            manifest['metadata']['file_structure']['fragments_start'] = fragments_start
            manifest['metadata']['file_structure']['fragments_size'] = fragments_size

            # 5. Go back and write actual manifest with updated positions
            manifest['metadata']['file_structure']['manifest_start'] = manifest_start
            manifest['metadata']['file_structure']['manifest_size'] = current_position - manifest_start
            manifest['metadata']['file_structure']['total_file_size'] = current_position

            manifest_json = json.dumps(manifest, indent=2).encode('utf-8')
            actual_manifest_box = create_mp4_box('manf', manifest_json)

            # Overwrite placeholder with actual manifest
            f.seek(manifest_start)
            f.write(actual_manifest_box)

            # Pad with zeros if needed
            if len(actual_manifest_box) < len(manifest_placeholder):
                padding = b'\x00' * (len(manifest_placeholder) - len(actual_manifest_box))
                f.write(padding)

        # 6. Save artifacts
        self.save_artifacts(manifest, faiss_index, output_path)

        print(f"Successfully encoded {manifest['metadata']['total_vectors']} vectors to {output_path}")
        print(f"Created {len(fragments)} fragments")
        print(f"File structure:")
        for key, value in manifest["metadata"]["file_structure"].items():
            print(f"  {key}: {value}")

    def save_artifacts(self,
                       manifest: Dict[str, Any],
                       index: Optional[faiss.Index],
                       output_path: str) -> None:
        """
        Save manifest and Faiss index as separate files

        Args:
            manifest: Complete manifest dictionary
            index: Faiss index object
            output_path: Path to the main MP4 output file
        """
        # Save manifest
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to {manifest_path}")

        # Save Faiss index
        if index:
            index_path = output_path.replace('.mp4', '_faiss.index')
            faiss.write_index(index, index_path)
            print(f"Faiss index saved to {index_path}")