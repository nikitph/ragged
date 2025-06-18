"""
FAISS index management for semantic search
Handles embedding generation, index creation, and vector search
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm

from .config import get_default_config

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages embeddings, FAISS index, and metadata for fast retrieval"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IndexManager

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()

        # Initialize embedding model
        model_name = self.config["embedding"]["model"]
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.config["embedding"]["dimension"]

        # Initialize FAISS index
        self.index = self._create_index()

        # Metadata storage
        self.metadata = []
        self.chunk_to_frame = {}  # Maps chunk ID to frame number
        self.frame_to_chunks = {}  # Maps frame number to chunk IDs

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        index_type = self.config["index"]["type"]

        if index_type == "Flat":
            # Exact search - best quality, works with any dataset size
            index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created Flat index for exact search")
        elif index_type == "IVF":
            # Inverted file index - faster for large datasets but needs training
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = self.config["index"]["nlist"]
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"Created IVF index with {nlist} clusters")
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add ID mapping for retrieval
        index = faiss.IndexIDMap(index)
        return index

    def add_chunks(self, chunks: List[str], frame_numbers: List[int],
                   show_progress: bool = True) -> List[int]:
        """
        Add chunks to index with embeddings

        Args:
            chunks: List of text chunks
            frame_numbers: Corresponding frame numbers for each chunk
            show_progress: Show progress bar

        Returns:
            List of successfully added chunk IDs
        """
        if len(chunks) != len(frame_numbers):
            raise ValueError("Number of chunks must match number of frame numbers")

        logger.info(f"Processing {len(chunks)} chunks for indexing...")

        # Filter valid chunks
        valid_chunks = []
        valid_frames = []
        skipped_count = 0

        for chunk, frame_num in zip(chunks, frame_numbers):
            if self._is_valid_chunk(chunk):
                valid_chunks.append(chunk)
                valid_frames.append(frame_num)
            else:
                skipped_count += 1
                logger.warning(f"Skipping invalid chunk at frame {frame_num}")

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid chunks")

        if not valid_chunks:
            logger.error("No valid chunks to process")
            return []

        # Generate embeddings
        try:
            embeddings = self._generate_embeddings(valid_chunks, show_progress)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        # Add to FAISS index
        try:
            chunk_ids = self._add_to_index(embeddings, valid_chunks, valid_frames)
            logger.info(f"Successfully added {len(chunk_ids)} chunks to index")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            return []

    def _is_valid_chunk(self, chunk: str) -> bool:
        """Validate chunk for embedding processing"""
        if not isinstance(chunk, str):
            return False

        chunk = chunk.strip()

        if len(chunk) == 0:
            return False

        if len(chunk) > 8192:  # SentenceTransformer limit
            return False

        # Check UTF-8 encoding
        try:
            chunk.encode('utf-8')
            return True
        except UnicodeEncodeError:
            return False

    def _generate_embeddings(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings with error handling"""
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=show_progress,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return np.array(embeddings).astype('float32')
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Try smaller batches as fallback
            return self._generate_embeddings_batched(chunks, show_progress)

    def _generate_embeddings_batched(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings in smaller batches"""
        all_embeddings = []
        batch_size = 50

        if show_progress:
            chunks_iter = tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings")
        else:
            chunks_iter = range(0, len(chunks), batch_size)

        for i in chunks_iter:
            batch = chunks[i:i + batch_size]
            try:
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    batch_size=16,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to process batch {i // batch_size}: {e}")
                # Skip this batch
                continue

        if not all_embeddings:
            raise RuntimeError("No embeddings could be generated")

        return np.array(all_embeddings).astype('float32')

    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str],
                      frame_numbers: List[int]) -> List[int]:
        """Add embeddings to FAISS index with training if needed"""

        # Assign chunk IDs
        start_id = len(self.metadata)
        chunk_ids = list(range(start_id, start_id + len(chunks)))

        # Train index if needed (for IVF)
        try:
            self._train_index_if_needed(embeddings)
        except Exception as e:
            logger.warning(f"Index training failed: {e}")
            logger.info("Falling back to Flat index")
            # Fallback to flat index
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

        # Add embeddings to index
        try:
            self.index.add_with_ids(embeddings, np.array(chunk_ids, dtype=np.int64))
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            raise

        # Store metadata
        for chunk, frame_num, chunk_id in zip(chunks, frame_numbers, chunk_ids):
            metadata = {
                "id": chunk_id,
                "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,  # Preview
                "frame": frame_num,
                "length": len(chunk)
            }
            self.metadata.append(metadata)

            # Update mappings
            self.chunk_to_frame[chunk_id] = frame_num
            if frame_num not in self.frame_to_chunks:
                self.frame_to_chunks[frame_num] = []
            self.frame_to_chunks[frame_num].append(chunk_id)

        return chunk_ids

    def _train_index_if_needed(self, embeddings: np.ndarray):
        """Train IVF index if required"""
        underlying_index = self.index.index

        if isinstance(underlying_index, faiss.IndexIVFFlat):
            if not underlying_index.is_trained:
                nlist = underlying_index.nlist

                if len(embeddings) < nlist:
                    raise RuntimeError(f"Insufficient data for IVF training: need {nlist}, got {len(embeddings)}")

                logger.info(f"Training IVF index with {len(embeddings)} vectors")
                underlying_index.train(embeddings)
                logger.info("IVF training completed")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar chunks

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk_id, distance, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        # Gather results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                metadata = self.metadata[idx]
                results.append((idx, float(dist), metadata))

        return results

    def get_chunks_by_frame(self, frame_number: int) -> List[Dict[str, Any]]:
        """Get all chunks associated with a frame"""
        chunk_ids = self.frame_to_chunks.get(frame_number, [])
        return [self.metadata[chunk_id] for chunk_id in chunk_ids if chunk_id < len(self.metadata)]

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID"""
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None

    def save(self, path: str):
        """
        Save index to disk

        Args:
            path: Path to save index (without extension)
        """
        path = Path(path)

        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))

        # Save metadata and mappings
        data = {
            "metadata": self.metadata,
            "chunk_to_frame": self.chunk_to_frame,
            "frame_to_chunks": self.frame_to_chunks,
            "config": self.config
        }

        with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved index to {path}")

    def load(self, path: str):
        """
        Load index from disk

        Args:
            path: Path to load index from (without extension)
        """
        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))

        # Load metadata and mappings
        with open(path.with_suffix('.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.metadata = data["metadata"]
        self.chunk_to_frame = {int(k): v for k, v in data["chunk_to_frame"].items()}
        self.frame_to_chunks = {int(k): v for k, v in data["frame_to_chunks"].items()}

        # Update config if available
        if "config" in data:
            self.config.update(data["config"])

        logger.info(f"Loaded index from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_chunks": len(self.metadata),
            "total_frames": len(self.frame_to_chunks),
            "index_type": self.config["index"]["type"],
            "embedding_model": self.config["embedding"]["model"],
            "dimension": self.dimension,
            "avg_chunks_per_frame": np.mean(
                [len(chunks) for chunks in self.frame_to_chunks.values()]) if self.frame_to_chunks else 0
        }