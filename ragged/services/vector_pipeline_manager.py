import logging
from typing import List, Dict

import numpy as np

from ragged.services.FaissIndexBuilder import FaissIndexBuilder
from ragged.services.TextProcessor import TextProcessor
from ragged.services.vector_clusterer import VectorClusterer
from ragged.services.vector_mp4_writer import VectorMP4Writer

# In refactored_pipeline.py, a new manager

# Make sure all necessary classes are imported
from tqdm import tqdm


class VectorPipelineManager:
    """
    Orchestrates the hierarchical pipeline to achieve both document integrity
    and IVFPQ search efficiency.
    """

    def __init__(self, model_name: str, chunk_size: int, chunk_overlap: int, n_clusters_per_doc: int):
        self.text_processor = TextProcessor(model_name, chunk_size, chunk_overlap)
        self.vector_clusterer = VectorClusterer(n_clusters_per_doc)
        self.faiss_builder = FaissIndexBuilder()
        self.mp4_writer = VectorMP4Writer(vector_dim=self.text_processor.vector_dim)

    def run_hierarchical(self, documents: List[Dict[str, str]], output_path: str):
        """
        Executes the hierarchical pipeline.
        - Each document is one MP4 fragment.
        - A global IVFPQ index is built from intra-document clusters.
        """
        logging.info("--- Starting Hierarchical Vector Pipeline ---")

        all_reordered_vectors = []
        all_centroids = []
        grouped_data = []  # Will hold final data for the writer
        article_id_counter = 0

        logging.info("Step 1: Processing and clustering each document individually...")
        for doc in tqdm(documents, desc="Processing & Clustering Articles"):
            vectors, chunks = self.text_processor.process_document(doc)
            if vectors.size == 0:
                continue

            # Cluster the vectors of this single document
            reordered_doc_vectors, doc_centroids, reorder_map = self.vector_clusterer.cluster_document_vectors(vectors)

            # Reorder the metadata to match the vectors
            reordered_chunks = [chunks[i] for i in reorder_map]

            all_reordered_vectors.append(reordered_doc_vectors)
            all_centroids.append(doc_centroids)

            grouped_data.append({
                "id": article_id_counter,
                "vectors": reordered_doc_vectors,
                "metadata": reordered_chunks
            })
            article_id_counter += 1

        if not all_reordered_vectors:
            logging.error("Pipeline stopped: No vectors were generated.")
            return

        # Concatenate all vectors and centroids for the global index
        global_vectors = np.concatenate(all_reordered_vectors, axis=0)
        global_centroids = np.concatenate(all_centroids, axis=0)
        logging.info(f"Total vectors: {len(global_vectors)}, Total sub-clusters (centroids): {len(global_centroids)}")

        # Step 2: Build the global hierarchical Faiss index
        faiss_index = self.faiss_builder.build_ivfpq_from_centroids(global_vectors, global_centroids)

        # Step 3: Write the final artifacts
        # We can reuse the writer from the previous refactoring
        self.mp4_writer.write_from_grouped_data(
            output_path,
            grouped_data,
            faiss_index
        )

        logging.info("--- Hierarchical Pipeline Finished Successfully ---")