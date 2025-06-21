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

    def process_documents(self, documents: List[Dict[str, str]]) -> Tuple[np.ndarray, List[Dict]]:
        """Process multiple documents into vectors"""
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')

            if not text.strip():
                continue

            chunks = self.chunk_text(text, source)
            all_chunks.extend(chunks)

        return self.encode_chunks(all_chunks)

    def process_text_files(self, file_paths: List[str]) -> Tuple[np.ndarray, List[Dict]]:
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
    """Enhanced encoder with CDN optimization features"""

    def __init__(self, vector_dim: int = 384, chunk_size: int = 1000):
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.fragments = []
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
                "file_structure": {
                    "ftyp_start": 0,
                    "ftyp_size": 0,
                    "video_track_start": 0,
                    "video_track_size": 0,
                    "manifest_start": 0,
                    "manifest_size": 0,
                    "fragments_start": 0,
                    "total_file_size": 0
                }
            },
            "vector_map": {}
        }

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
                # Byte positions will be filled during encoding
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

    def _build_faiss_index(self):
        """Build Faiss index for vector search"""
        if not self.all_vectors:
            return None

        vectors_array = np.array(self.all_vectors).astype(np.float32)
        n_vectors = len(vectors_array)

        faiss.normalize_L2(vectors_array)

        if n_vectors < 10000:
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
        """Serialize a fragment into binary format"""
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
        """Enhanced encode to MP4 with precise byte tracking"""
        print("Building Faiss index...")
        self.faiss_index = self._build_faiss_index()

        # Create all components first to calculate sizes
        ftyp_data = b'isom' + struct.pack('>I', 512) + b'isommp41avc1'
        ftyp_box = self._create_mp4_box('ftyp', ftyp_data)

        video_boxes = self._create_video_boxes()

        # Serialize all fragments first to get accurate sizes
        serialized_fragments = []
        fragment_boxes = []

        for i, fragment in enumerate(self.fragments):
            fragment_data = self._serialize_fragment(fragment)
            moof_header = struct.pack('>II', fragment["id"], fragment["vector_count"])
            moof_box = self._create_mp4_box('moof', moof_header)
            mdat_box = self._create_mp4_box('mdat', fragment_data)

            serialized_fragments.append({
                'moof': moof_box,
                'mdat': mdat_box,
                'data': fragment_data
            })
            fragment_boxes.extend([moof_box, mdat_box])

        # Update manifest with calculated byte positions
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

        # Manifest box (we'll update this after creating it)
        manifest_start = self.current_byte_position

        # Calculate fragments start position (after manifest)
        # We need to estimate manifest size first
        temp_manifest_data = json.dumps(self.manifest, indent=2).encode('utf-8')
        estimated_manifest_box = self._create_mp4_box('manf', temp_manifest_data)
        manifest_size = len(estimated_manifest_box)

        self.manifest["metadata"]["file_structure"]["manifest_start"] = manifest_start
        self.manifest["metadata"]["file_structure"]["manifest_size"] = manifest_size
        self.current_byte_position += manifest_size

        # Fragments start
        self.manifest["metadata"]["file_structure"]["fragments_start"] = self.current_byte_position

        # Update fragment byte positions
        for i, serialized_frag in enumerate(serialized_fragments):
            fragment_start = self.current_byte_position
            moof_size = len(serialized_frag['moof'])
            mdat_size = len(serialized_frag['mdat'])
            data_size = len(serialized_frag['data'])

            self.manifest["metadata"]["fragments"][i].update({
                "byte_start": fragment_start,
                "byte_end": fragment_start + moof_size + mdat_size,
                "byte_size": moof_size + mdat_size,
                "moof_start": fragment_start,
                "mdat_start": fragment_start + moof_size,
                "data_start": fragment_start + moof_size + 8,  # Skip mdat header (8 bytes)
                "data_size": data_size
            })

            self.current_byte_position += moof_size + mdat_size

        # Final file size
        self.manifest["metadata"]["file_structure"]["total_file_size"] = self.current_byte_position

        # Create final manifest with accurate byte positions
        final_manifest_data = json.dumps(self.manifest, indent=2).encode('utf-8')
        manifest_box = self._create_mp4_box('manf', final_manifest_data)

        # Write the complete MP4 file
        with open(output_path, 'wb') as f:
            # Write in order and track actual positions
            actual_position = 0

            # FTYP box
            f.write(ftyp_box)
            actual_position += len(ftyp_box)
            print(f"FTYP written: {len(ftyp_box)} bytes, position now: {actual_position}")

            # Video boxes
            for i, box in enumerate(video_boxes):
                f.write(box)
                actual_position += len(box)
                print(f"Video box {i} written: {len(box)} bytes, position now: {actual_position}")

            # Manifest box
            f.write(manifest_box)
            actual_position += len(manifest_box)
            print(f"Manifest written: {len(manifest_box)} bytes, position now: {actual_position}")

            # Fragment boxes with actual position tracking
            for i, serialized_frag in enumerate(serialized_fragments):
                fragment_actual_start = actual_position

                # Write moof
                f.write(serialized_frag['moof'])
                actual_position += len(serialized_frag['moof'])
                mdat_actual_start = actual_position

                # Write mdat
                f.write(serialized_frag['mdat'])
                actual_position += len(serialized_frag['mdat'])

                # Update manifest with actual positions
                self.manifest["metadata"]["fragments"][i].update({
                    "byte_start": fragment_actual_start,
                    "byte_end": actual_position,
                    "moof_start": fragment_actual_start,
                    "mdat_start": mdat_actual_start,
                    "data_start": mdat_actual_start + 8,  # Skip mdat header (8 bytes)
                    "data_size": len(serialized_frag['data'])
                })

                print(
                    f"Fragment {i} written: moof at {fragment_actual_start}, mdat at {mdat_actual_start}, data at {mdat_actual_start + 8}, size {len(serialized_frag['data'])}")

        # Update final file size
        self.manifest["metadata"]["file_structure"]["total_file_size"] = actual_position

        # Save separate manifest file for CDN optimization (with corrected positions)
        manifest_path = output_path.replace('.mp4', '_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        # Save Faiss index
        if self.faiss_index:
            faiss_path = output_path.replace('.mp4', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"Faiss index saved to {faiss_path}")

        print(f"Encoded {self.manifest['metadata']['total_vectors']} vectors to {output_path}")
        print(f"Created {len(self.fragments)} fragments")
        print(f"Manifest saved to {manifest_path}")
        print(f"File structure:")
        for key, value in self.manifest["metadata"]["file_structure"].items():
            print(f"  {key}: {value}")


# Example usage
def create_text_vector_mp4(documents: List[Dict[str, str]], output_path: str):
    """Complete pipeline: text documents → vectors → MP4 with CDN optimizations"""

    # Initialize pipeline
    pipeline = TextVectorPipeline(
        model_name="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        vector_dim=384
    )

    # Process documents into vectors
    vectors, metadata = pipeline.process_documents(documents)

    if len(vectors) == 0:
        print("No vectors generated from documents")
        return

    # Initialize enhanced MP4 encoder
    encoder = VectorMP4Encoder(vector_dim=384, chunk_size=1000)

    # Add vectors to encoder
    encoder.add_vectors(vectors, metadata)

    # Encode to MP4 with byte tracking
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

    create_text_vector_mp4(sample_docs, "knowledge_base.mp4")