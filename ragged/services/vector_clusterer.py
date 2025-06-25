# In refactored_pipeline.py, class VectorClusterer
from typing import Tuple

import faiss
import numpy as np


class VectorClusterer:
    """
    Performs K-means clustering. This version is adapted to cluster
    vectors within a single document to find its sub-topics.
    """

    def __init__(self, n_clusters_per_doc: int):
        """
        Args:
            n_clusters_per_doc: The number of sub-clusters to find within each document.
        """
        if n_clusters_per_doc <= 0:
            raise ValueError("Number of clusters must be greater than 0.")
        self.n_clusters_per_doc = n_clusters_per_doc

    def cluster_document_vectors(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Clusters vectors from a single document and reorders them.
        Handles cases with few vectors gracefully.
        """
        num_vectors, dim = vectors.shape

        # --- FIX 1: Dynamically adjust k ---
        # If we have fewer vectors than desired clusters, reduce k.
        n_clusters = min(self.n_clusters_per_doc, num_vectors)
        if n_clusters <= 1:  # No point clustering 1 or 0 items
            return vectors, vectors, np.arange(num_vectors)  # Return vectors as their own centroids

        kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=20, verbose=False)

        # --- FIX 2: Set minimum points per centroid to avoid warnings ---
        # We know we're working with small sets, so this is acceptable.
        kmeans.min_points_per_centroid = 1
        # Optionally, you could set it to a slightly higher number, like 2 or 3,
        # to ensure centroids aren't based on a single outlier. 1 is fine to start.

        kmeans.train(vectors)

        centroids = kmeans.centroids
        _, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()

        # Group original indices by local cluster
        cluster_members = [[] for _ in range(n_clusters)]
        for i, cluster_id in enumerate(assignments):
            cluster_members[cluster_id].append(i)

        reordered_indices = []
        for i in range(n_clusters):
            members = cluster_members[i]
            if not members:
                continue

            # Order members by similarity to their local centroid
            member_vectors = vectors[members]
            centroid = centroids[i]
            similarities = np.dot(member_vectors, centroid)
            sorted_member_indices = [m for _, m in sorted(zip(similarities, members), key=lambda p: p[0], reverse=True)]
            reordered_indices.extend(sorted_member_indices)

        reordered_vectors = vectors[reordered_indices]

        # Return reordered vectors, the centroids, and the mapping of old indices to new
        return reordered_vectors, centroids, np.array(reordered_indices)