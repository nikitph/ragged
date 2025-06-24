import hashlib
import re

import numpy as np
import tiktoken
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from ragged.models.text_chunk import TextChunk


class TextVectorPipeline:
    """Pipeline to convert text documents into vectors for MP4 encoding"""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 vector_dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_dim = vector_dim
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Verify model dimension
        test_vector = self.model.encode(["test"])
        if test_vector.shape[1] != vector_dim:
            self.vector_dim = test_vector.shape[1]

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _extract_topic(self, text: str) -> Optional[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        if not words:
            return None
        return max(set(words), key=words.count)

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        tokens = self.encoder.encode(text)
        return self.encoder.decode(tokens[-overlap_tokens:]) if len(tokens) > overlap_tokens else text

    def chunk_text(self, text: str, source: str = "unknown") -> List[TextChunk]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    topic=self._extract_topic(current_chunk),
                    word_count=len(current_chunk.split()),
                    token_count=current_tokens
                ))
                current_chunk = self._get_overlap_text(current_chunk, self.chunk_overlap) + " " + sentence
                current_tokens = self._count_tokens(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens

        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                topic=self._extract_topic(current_chunk),
                word_count=len(current_chunk.split()),
                token_count=current_tokens
            ))
        return chunks

    def encode_chunks(self, chunks: List[TextChunk]) -> Tuple[np.ndarray, List[Dict]]:
        texts = [chunk.text for chunk in chunks]
        vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        metadata = [{
            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "topic": chunk.topic,
            "word_count": chunk.word_count,
            "token_count": chunk.token_count,
            "timestamp": chunk.timestamp,
            "text_hash": hashlib.md5(chunk.text.encode()).hexdigest()[:8]
        } for chunk in chunks]

        return vectors, metadata

    def process_documents(self, documents: List[Dict[str, str]]) -> Tuple[np.ndarray, List[Dict]]:
        all_chunks = []
        for doc in documents:
            if text := doc.get('text', '').strip():
                all_chunks.extend(self.chunk_text(text, doc.get('source', 'unknown')))
        return self.encode_chunks(all_chunks)