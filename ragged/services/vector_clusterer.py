import logging
from typing import List, Dict, Tuple

import faiss
import numpy as np

from ragged.models.text_chunk import TextChunk


class VectorClusterer:
    """
    Performs K-means clustering and reorders vectors based on cluster
    and similarity to the cluster's centroid.
    """

    def __init__(self, n_clusters: int):
        """
        Args:
            n_clusters: The number of clusters to form.
        """
        if n_clusters <= 0:
            raise ValueError("Number of clusters must be greater than 0.")
        self.n_clusters = n_clusters

    def cluster_and_reorder(self, vectors: np.ndarray, metadata: List[TextChunk]
                            ) -> Tuple[np.ndarray, List[TextChunk], List[Dict]]:
        """
        Clusters vectors and reorders them.

        Args:
            vectors: A numpy array of vectors.
            metadata: A list of TextChunk objects.

        Returns:
            A tuple containing:
            - The reordered vector array.
            - The reordered list of TextChunks.
            - A list of dictionaries with info about each cluster (ID, size, etc.).
        """
        num_vectors, dim = vectors.shape

        if num_vectors < self.n_clusters:
            logging.warning(f"Number of vectors ({num_vectors}) is less than n_clusters ({self.n_clusters})."
                            f" Adjusting n_clusters to {num_vectors}.")
            self.n_clusters = num_vectors

        if self.n_clusters == 0:
            return vectors, metadata, []

        logging.info(f"Clustering {num_vectors} vectors into {self.n_clusters} clusters using Faiss K-means...")

        kmeans = faiss.Kmeans(d=dim, k=self.n_clusters, niter=20, verbose=False)
        kmeans.train(vectors)

        centroids = kmeans.centroids
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()

        # Group original indices by cluster
        cluster_members = [[] for _ in range(self.n_clusters)]
        for i, cluster_id in enumerate(assignments):
            cluster_members[cluster_id].append(i)

        reordered_indices = []
        cluster_info = []
        current_pos = 0

        logging.info("Reordering vectors based on cluster and similarity to centroid...")
        for i in range(self.n_clusters):
            members = cluster_members[i]
            if not members:
                continue

            # Get vectors and centroid for this cluster
            member_vectors = vectors[members]
            centroid = centroids[i]

            # Calculate similarity (dot product, since vectors are normalized)
            similarities = np.dot(member_vectors, centroid)

            # Sort members by similarity in descending order
            sorted_member_indices = [m for _, m in sorted(zip(similarities, members), key=lambda p: p[0], reverse=True)]

            reordered_indices.extend(sorted_member_indices)

            cluster_info.append({
                "cluster_id": i,
                "start_index": current_pos,
                "vector_count": len(sorted_member_indices)
            })
            current_pos += len(sorted_member_indices)

        # Apply the new order
        reordered_vectors = vectors[reordered_indices]
        reordered_metadata = [metadata[i] for i in reordered_indices]

        logging.info("Finished clustering and reordering.")
        return reordered_vectors, reordered_metadata, cluster_info