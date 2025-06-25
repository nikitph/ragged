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


# In refactored_pipeline.py, a modified FaissIndexBuilder

class FaissIndexBuilder:
    """Builds a Faiss index, specialized for the hierarchical structure."""

    def build_ivfpq_from_centroids(self, all_vectors: np.ndarray, all_centroids: np.ndarray) -> faiss.Index:
        """
        Builds an IndexIVFPQ where the quantizer is trained on the centroids
        of the intra-document clusters.

        Args:
            all_vectors: All reordered vectors from all documents, concatenated.
            all_centroids: All centroids from all documents, concatenated.

        Returns:
            A trained faiss.IndexIVFPQ object.
        """
        num_centroids, dim = all_centroids.shape
        nlist = num_centroids  # Each centroid defines a list in our index

        # PQ parameters
        m = 8 if dim % 8 == 0 else 4
        nbits = 8

        logging.info(f"Building hierarchical Faiss IndexIVFPQ: nlist={nlist}, m={m}, nbits={nbits}")

        # The quantizer is an index of the centroids themselves
        quantizer = faiss.IndexFlatIP(dim)
        quantizer.add(all_centroids)

        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        logging.info("Training IVFPQ index on all vectors...")
        index.train(all_vectors)  # Train the PQ part on the actual data

        logging.info("Adding vectors to the hierarchical index...")
        index.add(all_vectors)

        index.nprobe = min(nlist, 16)
        logging.info(f"Faiss index built successfully. nprobe set to {index.nprobe}.")
        return index