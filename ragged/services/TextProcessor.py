import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import os
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken
from dataclasses import dataclass, asdict
import re
from datetime import datetime
import logging

from ragged.models.text_chunk import TextChunk


class TextProcessor:
    """
    Handles the ingestion of raw text and conversion into text chunks and vector embeddings.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Args:
            model_name: SentenceTransformer model name.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Token overlap between chunks.
        """
        logging.info(f"Initializing TextProcessor with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
        logging.info(f"Model vector dimension: {self.vector_dim}")

    def _count_tokens(self, text: str) -> int:
        """Counts tokens in text using tiktoken."""
        return len(self.encoder.encode(text))

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Gets the last N tokens from text for overlap."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        overlap_token_ids = tokens[-overlap_tokens:]
        return self.encoder.decode(overlap_token_ids)

    def _chunk_document(self, text: str, source: str) -> List[TextChunk]:
        """Splits a single document's text into overlapping chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk_text = ""
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_text:
                chunks.append(TextChunk(
                    text=current_chunk_text.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    word_count=len(current_chunk_text.split()),
                    token_count=self._count_tokens(current_chunk_text),
                    timestamp=datetime.now().isoformat()
                ))
                overlap = self._get_overlap_text(current_chunk_text, self.chunk_overlap)
                current_chunk_text = overlap + " " + sentence
                chunk_id += 1
            else:
                current_chunk_text += (" " + sentence) if current_chunk_text else sentence
            current_tokens = self._count_tokens(current_chunk_text)

        if current_chunk_text.strip():
            chunks.append(TextChunk(
                text=current_chunk_text.strip(),
                source=source,
                chunk_id=chunk_id,
                word_count=len(current_chunk_text.split()),
                token_count=self._count_tokens(current_chunk_text),
                timestamp=datetime.now().isoformat()
            ))
        return chunks

    def process_documents(self, documents: List[Dict[str, str]]) -> Tuple[np.ndarray, List[TextChunk]]:
        """
        Processes multiple documents into a flat list of TextChunks and their embeddings.

        Args:
            documents: A list of dictionaries, each with 'text' and 'source' keys.

        Returns:
            A tuple containing:
            - A numpy array of vector embeddings.
            - A list of TextChunk objects corresponding to each vector.
        """
        logging.info(f"Processing {len(documents)} documents...")
        all_chunks = []
        for doc in documents:
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            if text.strip():
                all_chunks.extend(self._chunk_document(text, source))

        if not all_chunks:
            logging.warning("No text chunks were generated from the documents.")
            return np.array([]), []

        texts_to_encode = [chunk.text for chunk in all_chunks]
        logging.info(f"Encoding {len(texts_to_encode)} text chunks into vectors...")

        vectors = self.model.encode(
            texts_to_encode,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(vectors)  # Normalize for dot product similarity

        logging.info("Finished creating embeddings.")
        return vectors, all_chunks