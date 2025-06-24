import logging
from typing import List, Dict

from ragged.services.FaissIndexBuilder import FaissIndexBuilder
from ragged.services.TextProcessor import TextProcessor
from ragged.services.vector_clusterer import VectorClusterer
from ragged.services.vector_mp4_writer import VectorMP4Writer


class VectorPipelineManager:
    """
    Orchestrates the entire pipeline from text processing to MP4 writing.
    """

    def __init__(self, model_name: str, chunk_size: int, chunk_overlap: int, n_clusters: int):
        self.text_processor = TextProcessor(model_name, chunk_size, chunk_overlap)
        self.vector_clusterer = VectorClusterer(n_clusters)
        self.faiss_builder = FaissIndexBuilder()
        self.mp4_writer = VectorMP4Writer(vector_dim=self.text_processor.vector_dim)
        self.n_clusters = n_clusters

    def run(self, documents: List[Dict[str, str]], output_path: str):
        """
        Executes the full pipeline.

        Args:
            documents: A list of dictionaries, each with 'text' and 'source'.
            output_path: The base path for the output files (e.g., 'knowledge.mp4').
        """
        logging.info("--- Starting Vector Pipeline ---")

        # Step 1: Ingest and embed text
        vectors, metadata = self.text_processor.process_documents(documents)
        if vectors.size == 0:
            logging.error("Pipeline stopped: No vectors were generated.")
            return

        # Step 2: Cluster and reorder vectors
        # Use the actual number of clusters determined by the clusterer
        effective_n_clusters = min(self.n_clusters, len(vectors))
        self.vector_clusterer.n_clusters = effective_n_clusters

        ordered_vectors, ordered_metadata, cluster_info = self.vector_clusterer.cluster_and_reorder(vectors, metadata)

        # Step 3: Build the Faiss index
        faiss_index = self.faiss_builder.build_index(ordered_vectors, effective_n_clusters)

        # Step 4: Write all artifacts to disk
        self.mp4_writer.write(output_path, ordered_vectors, ordered_metadata, cluster_info, faiss_index)

        logging.info("--- Vector Pipeline Finished Successfully ---")