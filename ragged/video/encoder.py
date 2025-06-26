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

from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

from ragged.services.uploader.r2_uploader import UploadServiceBuilder, R2Config


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

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # Approximate chars from tokens
            chunk_overlap=chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
            length_function=self._count_tokens
        )

        # Load spacy model for topic extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spacy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

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

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap + " " + sentence if overlap else sentence
                current_tokens = self._count_tokens(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens from text for overlap"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= overlap_tokens:
            return text

        overlap_token_ids = tokens[-overlap_tokens:]
        return self.encoder.decode(overlap_token_ids)

    def _extract_topic_smart(self, text: str) -> Optional[str]:
        """Extract topic using spacy NER or fallback method"""
        if self.nlp:
            # Use spacy for named entity recognition
            doc = self.nlp(text)

            # Prioritize organizations, products, technologies
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "TECH", "WORK_OF_ART"]:
                    return ent.text.lower()

            # Fall back to noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
            if noun_phrases:
                return noun_phrases[0]

        # Fallback: simple keyword extraction
        return self._extract_topic_basic(text)

    def _extract_topic_basic(self, text: str) -> Optional[str]:
        """Basic topic extraction using word frequency"""
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        stop_words = {'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
                      'will', 'would', 'could', 'should', 'this', 'that', 'with', 'from'}
        filtered = [w for w in words if w not in stop_words]

        if filtered:
            word_freq = {}
            for word in filtered:
                word_freq[word] = word_freq.get(word, 0) + 1
            return max(word_freq.items(), key=lambda x: x[1])[0]

        return None

    def encode_chunks(self, chunks: List[TextChunk]) -> Tuple[np.ndarray, List[Dict]]:
        """Convert text chunks to vectors with full text storage"""
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