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

class FaissIndexBuilder:
    """
    Builds a Faiss IndexIVFPQ from a set of vectors.
    """
    def build_index(self, vectors: np.ndarray, n_clusters: int) -> faiss.Index:
        """
        Builds the index. The nlist parameter is set to n_clusters.

        Args:
            vectors: The final, ordered numpy array of vectors.
            n_clusters: The number of clusters, which will be used as nlist.

        Returns:
            A trained faiss.IndexIVFPQ object.
        """
        if n_clusters == 0 or len(vectors) == 0:
            logging.warning("Cannot build Faiss index with 0 vectors or 0 clusters.")
            # Return a dummy empty index
            return faiss.IndexFlatIP(vectors.shape[1])

        num_vectors, dim = vectors.shape
        nlist = n_clusters

        # Choose PQ parameters (m: sub-quantizers, nbits: bits per code)
        m = 8 if dim % 8 == 0 else 4 if dim % 4 == 0 else 1
        nbits = 8

        logging.info(f"Building Faiss IndexIVFPQ for {num_vectors} vectors: nlist={nlist}, m={m}, nbits={nbits}")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        logging.info("Training IVFPQ index...")
        index.train(vectors)

        logging.info("Adding vectors to index...")
        index.add(vectors)

        # Set nprobe for search-time performance
        index.nprobe = min(nlist, 16)

        logging.info(f"Faiss index built successfully. nprobe set to {index.nprobe}.")
        return index