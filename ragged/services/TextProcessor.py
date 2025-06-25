# In your pipeline file (e.g., refactored_pipeline.py)
import logging
import re
from datetime import datetime
from typing import List, Dict, Tuple

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from ragged.models.text_chunk import TextChunk


# from .data_models import TextChunk # Your import might vary

class TextProcessor:
    # ... (__init__ is the same) ...
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        logging.info(f"Initializing TextProcessor with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Use a specific encoding for consistency
        self.encoder = tiktoken.get_encoding("cl100k_base")
        logging.info(f"Model vector dimension: {self.vector_dim}")

    ### --- FIX: Replaced sentence-based chunking with token-based chunking --- ###
    def _chunk_document(self, text: str, source: str) -> List[TextChunk]:
        """
        Splits a document into NON-OVERLAPPING chunks that respect sentence boundaries.
        """
        if not text.strip():
            return []

        # Split the text into sentences using a more robust regex
        # This regex splits on '.', '!', '?' followed by space, but keeps the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_token_count = len(self.encoder.encode(sentence))

            # If adding the next sentence exceeds the chunk size, finalize the current chunk
            if current_chunk_text and (current_chunk_tokens + sentence_token_count > self.chunk_size):
                chunks.append(TextChunk(
                    text=current_chunk_text.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    word_count=len(current_chunk_text.split()),
                    token_count=current_chunk_tokens,
                    timestamp=datetime.now().isoformat()
                ))
                # Start a new chunk
                current_chunk_text = ""
                current_chunk_tokens = 0
                chunk_id += 1

            # Add the sentence to the current chunk
            # Add a space if the chunk is not empty
            separator = " " if current_chunk_text else ""
            current_chunk_text += separator + sentence
            current_chunk_tokens += sentence_token_count + (1 if separator else 0)

        # Add the last remaining chunk if it exists
        if current_chunk_text:
            chunks.append(TextChunk(
                text=current_chunk_text.strip(),
                source=source,
                chunk_id=chunk_id,
                word_count=len(current_chunk_text.split()),
                token_count=current_chunk_tokens,
                timestamp=datetime.now().isoformat()
            ))

        return chunks

    # --- The rest of the class remains the same ---
    # It correctly uses the now-fixed _chunk_document method.
    def process_document(self, document: Dict[str, str]) -> Tuple[np.ndarray, List[TextChunk]]:
        """
        Processes a SINGLE document into a set of TextChunks and their embeddings.
        """
        text = document.get('text', '')
        source = document.get('source', 'unknown')

        # Use the new and improved chunking method
        chunks = self._chunk_document(text, source)

        if not chunks:
            return np.array([]), []

        texts_to_encode = [chunk.text for chunk in chunks]

        vectors = self.model.encode(
            texts_to_encode,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(vectors)

        return vectors, chunks