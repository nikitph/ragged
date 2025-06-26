import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import os
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken
from dataclasses import dataclass
import re
from datetime import datetime

from ragged.services.uploader.r2_uploader import UploadServiceBuilder, R2Config
from .knowledge_graph import KnowledgeGraphTrack, KnowledgeGraphIntegration


@dataclass
class TextChunk:
    """Represents a processed text chunk with metadata"""
    text: str
    source: str
    chunk_id: int
    topic: Optional[str] = None
    word_count: int = 0
    token_count: int = 0
    timestamp: str = ""


class TextVectorPipeline:
    """Pipeline to convert text documents into vectors for MP4 encoding"""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 vector_dim: int = 384):
        """
        Initialize the text-vector pipeline

        Args:
            model_name: SentenceTransformer model name
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            vector_dim: Vector dimension (must match model output)
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_dim = vector_dim
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Verify model dimension matches expected
        test_vector = self.model.encode(["test"])
        actual_dim = test_vector.shape[1]
        if actual_dim != vector_dim:
            print(f"Warning: Model outputs {actual_dim}D vectors, but expected {vector_dim}D")
            self.vector_dim = actual_dim

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoder.encode(text))

    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract topic from text using simple keyword analysis"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        if not words:
            return None

        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        if word_freq:
            return max(word_freq.items(), key=lambda x: x[1])[0]
        return None

    def chunk_text(self, text: str, source: str = "unknown") -> List[TextChunk]:
        """Split text into overlapping chunks based on token count"""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    topic=self._extract_topic(current_chunk),
                    word_count=len(current_chunk.split()),
                    token_count=current_tokens,
                    timestamp=datetime.now().isoformat()
                )
                chunks.append(chunk)

                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self._count_tokens(current_chunk)
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens

        if current_chunk.strip():
            chunk = TextChunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                topic=self._extract_topic(current_chunk),
                word_count=len(current_chunk.split()),
                token_count=current_tokens,
                timestamp=datetime.now().isoformat()
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens from text for overlap"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= overlap_tokens:
            return text

        overlap_token_ids = tokens[-overlap_tokens:]
        return self.encoder.decode(overlap_token_ids)

    def encode_chunks(self, chunks: List[TextChunk]) -> Tuple[np.ndarray, List[Dict]]:
        """Convert text chunks to vectors using the embedding model"""
        if not chunks:
            return np.array([]), []

        texts = [chunk.text for chunk in chunks]

        print(f"Encoding {len(texts)} text chunks...")
        vectors = self.model.encode(texts,
                                    show_progress_bar=True,
                                    convert_to_numpy=True)

        metadata = []
        for chunk in chunks:
            meta = {
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "topic": chunk.topic,
                "word_count": chunk.word_count,
                "token_count": chunk.token_count,
                "timestamp": chunk.timestamp,
                "text_hash": hashlib.md5(chunk.text.encode()).hexdigest()[:8]
            }
            metadata.append(meta)

        return vectors, metadata

    def process_documents(self, documents: List[Dict[str, str]]) -> Tuple[np.ndarray, List[Dict], List[TextChunk]]:
        """Process multiple documents into vectors and return chunks for KG processing"""
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')

            if not text.strip():
                continue

            chunks = self.chunk_text(text, source)
            all_chunks.extend(chunks)

        vectors, metadata = self.encode_chunks(all_chunks)
        return vectors, metadata, all_chunks

    def process_text_files(self, file_paths: List[str]) -> Tuple[np.ndarray, List[Dict], List[TextChunk]]:
        """Process text files directly"""
        documents = []

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append({
                        'text': text,
                        'source': os.path.basename(file_path)
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        return self.process_documents(documents)


class VectorMP4Encoder:
    """Enhanced encoder with CDN optimization features and Knowledge Graph support"""

    def __init__(self, vector_dim: int = 384, chunk_size: int = 1000, enable_kg: bool = True):
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.enable_kg = enable_kg
        self.fragments = []
        self.kg_fragments = []
        self.faiss_index = None
        self.all_vectors = []
        self.current_byte_position = 0
        self.manifest = {
            "metadata": {
                "vector_dim": vector_dim,
                "chunk_size": chunk_size,
                "total_vectors": 0,
                "fragments": [],
                "faiss_index_type": "IndexFlatIP",
                "knowledge_graph": {
                    "enabled": enable_kg,
                    "total_entities": 0,
                    "total_relations": 0,
                    "fragments": []
                },
                "file_structure": {
                    "ftyp_start": 0,
                    "ftyp_size": 0,
                    "video_track_start": 0,
                    "video_track_size": 0,
                    "vector_track_start": 0,
                    "vector_track_size": 0,
                    "kg_track_start": 0,
                    "kg_track_size": 0,
                    "manifest_start": 0,
                    "manifest_size": 0,
                    "total_file_size": 0
                }
            },
            "vector_map": {},
            "kg_map": {}
        }

        # Initialize KG components if enabled
        if self.enable_kg:
            self.kg_track = KnowledgeGraphTrack()
            self.kg_integration = KnowledgeGraphIntegration(self.kg_track)

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors with metadata to be encoded"""
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {vectors.shape[1]}")

        self.all_vectors.extend(vectors)

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

            frag_info = {
                "id": fragment_id,
                "vector_count": len(chunk_vectors),
                "start_idx": start_vector_id,
                "end_idx": end_vector_id,
                "topics": list(set([m.get("topic", "") for m in chunk_metadata if m.get("topic")])),
                "byte_start": 0,
                "byte_end": 0,
                "byte_size": 0,
                "moof_start": 0,
                "mdat_start": 0,
                "data_start": 0,
                "data_size": 0
            }
            self.manifest["metadata"]["fragments"].append(frag_info)

            for j, vec_meta in enumerate(chunk_metadata):
                vector_id = start_vector_id + j
                self.manifest["vector_map"][vector_id] = {
                    "fragment_id": fragment_id,
                    "local_offset": j,
                    "metadata": vec_meta
                }

        self.manifest["metadata"]["total_vectors"] += num_vectors

    def add_knowledge_graph_data(self, text_chunks: List[TextChunk]):
        """Process text chunks and add knowledge graph data"""
        if not self.enable_kg:
            print("Knowledge graph processing disabled")
            return

        print("Processing knowledge graph data...")

        # Process text chunks through KG pipeline
        kg_fragments_data, kg_metadata = self.kg_integration.process_and_integrate(text_chunks)

        # Store KG fragments
        for i, kg_fragment_data in enumerate(kg_fragments_data):
            kg_fragment = {
                "id": i,
                "data": kg_fragment_data,
                "size": len(kg_fragment_data)
            }
            self.kg_fragments.append(kg_fragment)

        # Update manifest with KG metadata
        self.manifest["metadata"]["knowledge_graph"].update(kg_metadata)

        # Add KG fragment info to manifest
        for i, kg_fragment in enumerate(self.kg_fragments):
            kg_frag_info = {
                "id": i,
                "size": kg_fragment["size"],
                "byte_start": 0,  # Will be updated during encoding
                "byte_end": 0,
                "moof_start": 0,
                "mdat_start": 0,
                "data_start": 0,
                "data_size": kg_fragment["size"]
            }
            self.manifest["metadata"]["knowledge_graph"]["fragments"].append(kg_frag_info)

            # Add to KG map
            self.manifest["kg_map"][i] = {
                "fragment_id": i,
                "size": kg_fragment["size"]
            }

    def _build_faiss_index(self):
        """Build Faiss index for vector search"""
        if not self.all_vectors:
            return None

        vectors_array = np.array(self.all_vectors).astype(np.float32)
        n_vectors = len(vectors_array)

        faiss.normalize_L2(vectors_array)

        if n_vectors < 100000:
            print(f"Using IndexFlatIP for {n_vectors} vectors")
            index = faiss.IndexFlatIP(self.vector_dim)
            index.add(vectors_array)
        else:
            nlist = min(int(4 * np.sqrt(n_vectors)), n_vectors // 10)
            nlist = max(nlist, 1)

            m = 8
            while self.vector_dim % m != 0 and m > 1:
                m -= 1

            nbits = 8

            print(f"Using IndexIVFPQ for {n_vectors} vectors: nlist={nlist}, m={m}, nbits={nbits}")

            quantizer = faiss.IndexFlatIP(self.vector_dim)
            index = faiss.IndexIVFPQ(quantizer, self.vector_dim, nlist, m, nbits)

            print("Training IVFPQ index...")
            index.train(vectors_array)
            index.add(vectors_array)
            index.nprobe = min(nlist, 10)

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
        """Serialize a vector fragment into binary format"""
        header = {
            "id": fragment["id"],
            "vector_count": fragment["vector_count"],
            "vector_dim": self.vector_dim,
            "start_idx": fragment["start_idx"],
            "end_idx": fragment["end_idx"]
        }

        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json)

        vectors_data = fragment["vectors"].astype(np.float32).tobytes()
        metadata_json = json.dumps(fragment["metadata"]).encode('utf-8')
        metadata_size = len(metadata_json)

        packed_data = (
                struct.pack('<I', header_size) + header_json +
                struct.pack('<I', len(vectors_data)) + vectors_data +
                struct.pack('<I', metadata_size) + metadata_json
        )

        return packed_data

    def _serialize_kg_fragment(self, kg_fragment: Dict) -> bytes:
        """Serialize a knowledge graph fragment into binary format"""
        # KG fragments are already serialized by the KG integration
        return kg_fragment["data"]

    def _create_minimal_video_track(self) -> bytes:
        """Create a minimal video track with single black frame"""
        h264_frame = bytes([
            0x00, 0x00, 0x00, 0x01,
            0x67, 0x42, 0x00, 0x0a,
            0x8d, 0x68, 0x05, 0x8b,
            0x00, 0x00, 0x00, 0x01,
            0x68, 0xce, 0x06, 0xe2,
            0x00, 0x00, 0x00, 0x01,
            0x65, 0x88, 0x80, 0x10,
            0x00, 0x02, 0x00, 0x08
        ])
        return h264_frame

    def _create_video_boxes(self) -> List[bytes]:
        """Create minimal video track boxes for MP4 compatibility"""
        duration = min(max(self.manifest["metadata"]["total_vectors"] // 100, 1), 60) * 1000

        mvhd_data = struct.pack('>IIIIII',
                                0, 0, 0, 1000, duration, 0x00010000
                                ) + b'\x00' * 76
        mvhd_box = self._create_mp4_box('mvhd', mvhd_data)

        tkhd_data = struct.pack('>IIIIII',
                                0x0000000F, 0, 0, 1, 0, duration
                                ) + b'\x00' * 60
        tkhd_box = self._create_mp4_box('tkhd', tkhd_data)

        mdhd_data = struct.pack('>IIIIII', 0, 0, 0, 1000, duration, 0)
        mdhd_box = self._create_mp4_box('mdhd', mdhd_data)

        hdlr_data = struct.pack('>IIII', 0, 0, 0x76696465, 0) + b'VideoHandler\x00'
        hdlr_box = self._create_mp4_box('hdlr', hdlr_data)

        mdia_content = mdhd_box + hdlr_box
        mdia_box = self._create_mp4_box('mdia', mdia_content)

        trak_content = tkhd_box + mdia_box
        trak_box = self._create_mp4_box('trak', trak_content)

        moov_content = mvhd_box + trak_box
        moov_box = self._create_mp4_box('moov', moov_content)

        video_frame = self._create_minimal_video_track()
        mdat_video_box = self._create_mp4_box('mdat', video_frame)

        return [moov_box, mdat_video_box]

    def _create_mp4_box(self, box_type: str, data: bytes) -> bytes:
        """Create MP4 box with type and data"""
        size = len(data) + 8
        return struct.pack('>I', size) + box_type.encode('ascii') + data

    def encode_to_mp4(self, output_path: str):
        """Enhanced encode to MP4 with vector and knowledge graph tracks"""
        print("Building Faiss index...")
        self.faiss_index = self._build_faiss_index()

        # Create all components first to calculate sizes
        ftyp_data = b'isom' + struct.pack('>I', 512) + b'isommp41avc1'
        ftyp_box = self._create_mp4_box('ftyp', ftyp_data)

        video_boxes = self._create_video_boxes()

        # Serialize all vector fragments
        serialized_vector_fragments = []
        vector_fragment_boxes = []

        for i, fragment in enumerate(self.fragments):
            fragment_data = self._serialize_fragment(fragment)
            moof_header = struct.pack('>II', fragment["id"], fragment["vector_count"])
            moof_box = self._create_mp4_box('moof', moof_header)
            mdat_box = self._create_mp4_box('mdat', fragment_data)

            serialized_vector_fragments.append({
                'moof': moof_box,
                'mdat': mdat_box,
                'data': fragment_data
            })
            vector_fragment_boxes.extend([moof_box, mdat_box])

        # Serialize all KG fragments
        serialized_kg_fragments = []
        kg_fragment_boxes = []

        if self.enable_kg and self.kg_fragments:
            for i, kg_fragment in enumerate(self.kg_fragments):
                kg_data = self._serialize_kg_fragment(kg_fragment)
                kg_moof_header = struct.pack('>II', kg_fragment["id"], len(kg_data))
                kg_moof_box = self._create_mp4_box('kgmf', kg_moof_header)  # Custom KG moof type
                kg_mdat_box = self._create_mp4_box('kgdt', kg_data)  # Custom KG mdat type

                serialized_kg_fragments.append({
                    'moof': kg_moof_box,
                    'mdat': kg_mdat_box,
                    'data': kg_data
                })
                kg_fragment_boxes.extend([kg_moof_box, kg_mdat_box])

        # Calculate byte positions
        self.current_byte_position = 0

        # FTYP box
        self.manifest["metadata"]["file_structure"]["ftyp_start"] = self.current_byte_position
        self.manifest["metadata"]["file_structure"]["ftyp_size"] = len(ftyp_box)
        self.current_byte_position += len(ftyp_box)

        # Video track boxes
        self.manifest["metadata"]["file_structure"]["video_track_start"] = self.current_byte_position
        video_track_size = sum(len(box) for box in video_boxes)
        self.manifest["metadata"]["file_structure"]["video_track_size"] = video_track_size
        self.current_byte_position += video_track_size

        # Vector track
        self.manifest["metadata"]["file_structure"]["vector_track_start"] = self.current_byte_position
        vector_track_size = sum(len(box) for box in vector_fragment_boxes)
        self.manifest["metadata"]["file_structure"]["vector_track_size"] = vector_track_size
        self.current_byte_position += vector_track_size

        # KG track
        self.manifest["metadata"]["file_structure"]["kg_track_start"] = self.current_byte_position
        kg_track_size = sum(len(box) for box in kg_fragment_boxes) if kg_fragment_boxes else 0
        self.manifest["metadata"]["file_structure"]["kg_track_size"] = kg_track_size
        self.current_byte_position += kg_track_size

        # Manifest position (estimate)
        manifest_start = self.current_byte_position
        temp_manifest_data = json.dumps(self.manifest, indent=2).encode('utf-8')
        estimated_manifest_box = self._create_mp4_box('manf', temp_manifest_data)
        manifest_size = len(estimated_manifest_box)

        self.manifest["metadata"]["file_structure"]["manifest_start"] = manifest_start
        self.manifest["metadata"]["file_structure"]["manifest_size"] = manifest_size
        self.current_byte_position += manifest_size

        # Final file size
        self.manifest["metadata"]["file_structure"]["total_file_size"] = self.current_byte_position

        # Update fragment byte positions
        current_pos = self.manifest["metadata"]["file_structure"]["vector_track_start"]
        for i, serialized_frag in enumerate(serialized_vector_fragments):
            fragment_start = current_pos
            moof_size = len(serialized_frag['moof'])
            mdat_size = len(serialized_frag['mdat'])
            data_size = len(serialized_frag['data'])

            self.manifest["metadata"]["fragments"][i].update({
                "byte_start": fragment_start,
                "byte_end": fragment_start + moof_size + mdat_size,
                "byte_size": moof_size + mdat_size,
                "moof_start": fragment_start,
                "mdat_start": fragment_start + moof_size,
                "data_start": fragment_start + moof_size + 8,
                "data_size": data_size
            })

            current_pos += moof_size + mdat_size

        # Update KG fragment byte positions
        current_pos = self.manifest["metadata"]["file_structure"]["kg_track_start"]
        for i, serialized_kg_frag in enumerate(serialized_kg_fragments):
            kg_fragment_start = current_pos
            kg_moof_size = len(serialized_kg_frag['moof'])
            kg_mdat_size = len(serialized_kg_frag['mdat'])
            kg_data_size = len(serialized_kg_frag['data'])

            self.manifest["metadata"]["knowledge_graph"]["fragments"][i].update({
                "byte_start": kg_fragment_start,
                "byte_end": kg_fragment_start + kg_moof_size + kg_mdat_size,
                "moof_start": kg_fragment_start,
                "mdat_start": kg_fragment_start + kg_moof_size,
                "data_start": kg_fragment_start + kg_moof_size + 8,
                "data_size": kg_data_size
            })

            current_pos += kg_moof_size + kg_mdat_size

        # Create final manifest with accurate byte positions
        final_manifest_data = json.dumps(self.manifest, indent=2).encode('utf-8')
        manifest_box = self._create_mp4_box('manf', final_manifest_data)

        # Write the complete MP4 file
        with open(output_path, 'wb') as f:
            # Write FTYP box
            f.write(ftyp_box)
            print(f"FTYP written: {len(ftyp_box)} bytes")

            # Write video boxes
            for i, box in enumerate(video_boxes):
                f.write(box)
                print(f"Video box {i} written: {len(box)} bytes")

            # Write vector fragment boxes
            for i, serialized_frag in enumerate(serialized_vector_fragments):
                f.write(serialized_frag['moof'])
                f.write(serialized_frag['mdat'])
                print(f"Vector fragment {i} written: {len(serialized_frag['data'])} bytes")

            # Write KG fragment boxes
            for i, serialized_kg_frag in enumerate(serialized_kg_fragments):
                f.write(serialized_kg_frag['moof'])
                f.write(serialized_kg_frag['mdat'])
                print(f"KG fragment {i} written: {len(serialized_kg_frag['data'])} bytes")

            # Write manifest box
            f.write(manifest_box)
            print(f"Manifest written: {len(manifest_box)} bytes")

        # Save separate manifest file
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        # Save Faiss index
        if self.faiss_index:
            faiss_path = output_path.replace('.mp4', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"Faiss index saved to {faiss_path}")

        print(f"Encoded {self.manifest['metadata']['total_vectors']} vectors to {output_path}")
        print(f"Created {len(self.fragments)} vector fragments")
        if self.enable_kg:
            print(f"Created {len(self.kg_fragments)} knowledge graph fragments")
            print(f"Extracted {self.manifest['metadata']['knowledge_graph']['total_entities']} entities")
            print(f"Extracted {self.manifest['metadata']['knowledge_graph']['total_relations']} relations")
        print(f"Manifest saved to {manifest_path}")

    def encode_and_upload(self, output_path: str):
        # Existing encoding logic
        self.encode_to_mp4(output_path)

        upload_service = UploadServiceBuilder.build()
        upload_service.execute_upload(output_path)


# Updated main function
def create_text_vector_mp4(documents: List[Dict[str, str]], output_path: str, enable_kg: bool = True):
    """Complete pipeline: text documents → vectors + knowledge graph → MP4"""

    # Initialize pipeline
    pipeline = TextVectorPipeline(
        model_name="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        vector_dim=384
    )

    # Process documents into vectors and get text chunks for KG processing
    vectors, metadata, text_chunks = pipeline.process_documents(documents)

    if len(vectors) == 0:
        print("No vectors generated from documents")
        return

    # Initialize enhanced MP4 encoder with KG support
    encoder = VectorMP4Encoder(vector_dim=384, chunk_size=1000, enable_kg=enable_kg)

    # Add vectors to encoder
    encoder.add_vectors(vectors, metadata)

    # Add knowledge graph data
    if enable_kg:
        encoder.add_knowledge_graph_data(text_chunks)

    # Encode to MP4 with both vector and KG tracks
    encoder.encode_to_mp4(output_path)


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
            "source": "ml_intro.txt"
        },
        {
            "text": "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign.",
            "source": "deep_learning.txt"
        },
        {
            "text": "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            "source": "nlp_basics.txt"
        }
    ]

    create_text_vector_mp4(sample_docs, "knowledge_base.mp4", enable_kg=True)