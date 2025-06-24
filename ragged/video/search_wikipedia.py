#!/usr/bin/env python3
"""
Wikipedia MP4 Vector Search CLI (Refactored & Corrected)

Search through a vector database created by the VectorPipelineManager.
This script reads the manifest, Faiss index, and MP4 file directly.
"""

import argparse
import json
import numpy as np
import faiss
import struct
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import sys
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class VectorSearchEngine:
    """
    Search engine for vector databases created by the VectorPipelineManager.
    It reads the manifest, Faiss index, and MP4 file to perform searches.
    """

    def __init__(self, mp4_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.mp4_path = mp4_path
        self.manifest_path = mp4_path.replace('.mp4', '_manifest.json')
        self.index_path = mp4_path.replace('.mp4', '_faiss.index')

        self._validate_paths()

        logging.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        logging.info(f"Loading manifest: {self.manifest_path}")
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        logging.info(f"Loading Faiss index: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        ### --- FIX: Build a cluster-to-vector map on initialization --- ###
        self.cluster_vector_map = self._build_cluster_map()

        # A cache to store deserialized metadata for a cluster to avoid re-reading the file
        self.cluster_cache = {}

        self._print_database_info()

    ### --- FIX: New method to pre-calculate all cluster start indices --- ###
    def _build_cluster_map(self) -> Dict[int, Dict]:
        """Builds a map from cluster_id to its vector start index and count."""
        logging.info("Building cluster-to-vector-index map...")
        cluster_map = {}
        # IMPORTANT: Sort fragments by cluster_id to calculate cumulative offsets correctly.
        sorted_fragments = sorted(
            self.manifest['metadata']['file_layout']['fragments'],
            key=lambda f: f['cluster_id']
        )

        current_vector_index = 0
        for fragment in sorted_fragments:
            cluster_id = fragment['cluster_id']
            vector_count = fragment['vector_count']
            cluster_map[cluster_id] = {
                'start_index': current_vector_index,
                'vector_count': vector_count,
                # Store the file layout info as well for easier access
                'file_layout': fragment
            }
            current_vector_index += vector_count

        # Sanity check to ensure map covers all vectors
        total_vectors_from_manifest = self.manifest['metadata']['total_vectors']
        if current_vector_index != total_vectors_from_manifest:
            logging.warning(
                f"Vector count mismatch! Map covers {current_vector_index} vectors, "
                f"but manifest reports {total_vectors_from_manifest}."
            )

        return cluster_map

    def _validate_paths(self):
        """Ensure all required files exist."""
        for path in [self.mp4_path, self.manifest_path, self.index_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

    def _print_database_info(self):
        """Display information about the loaded database from the manifest."""
        meta = self.manifest["metadata"]
        print("\n📊 Database Information:")
        print("=" * 50)
        print(f"📚 Total vectors: {meta['total_vectors']:,}")
        print(f"🎯 Vector dimensions: {meta['vector_dim']}")
        print(f"📦 Clusters (Fragments): {meta['total_clusters']}")
        print(f"🔍 Index type: {meta['faiss_index_type']}")
        print("=" * 50)

    ### --- FIX: Rewritten to use the pre-computed map --- ###
    def _find_cluster_for_vector_id(self, vector_id: int) -> Tuple[int, int]:
        """Find which cluster a global vector ID belongs to using the pre-built map."""
        for cluster_id, info in self.cluster_vector_map.items():
            start_index = info['start_index']
            count = info['vector_count']
            if start_index <= vector_id < start_index + count:
                local_index = vector_id - start_index
                return cluster_id, local_index
        raise ValueError(
            f"Vector ID {vector_id} not found in any cluster range. "
            f"Total vectors in map: {self.manifest['metadata']['total_vectors']}"
        )

    def _get_metadata_for_id(self, vector_id: int) -> Dict:
        """Fetches the metadata for a single vector ID by reading the MP4 file."""
        cluster_id, local_index = self._find_cluster_for_vector_id(vector_id)

        if cluster_id in self.cluster_cache:
            return self.cluster_cache[cluster_id][local_index]

        # Find fragment info from our pre-built map
        fragment_info = self.cluster_vector_map[cluster_id]['file_layout']

        with open(self.mp4_path, 'rb') as f:
            f.seek(fragment_info['mdat_offset'] + 8)  # +8 to skip mdat header

            vector_dim = self.manifest['metadata']['vector_dim']
            vector_count = fragment_info['vector_count']
            vector_bytes_size = vector_count * vector_dim * 4
            f.seek(vector_bytes_size, 1)

            metadata_size_bytes = f.read(4)
            metadata_size = struct.unpack('>I', metadata_size_bytes)[0]

            metadata_json = f.read(metadata_size).decode('utf-8')
            metadata_list = json.loads(metadata_json)

            self.cluster_cache[cluster_id] = metadata_list
            return metadata_list[local_index]

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not query.strip(): return []
        logging.info(f"Searching for: '{query}' with top_k={top_k}")

        query_vector = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vector)

        similarities, indices = self.index.search(query_vector, top_k)

        results = []
        if indices.size > 0:
            for sim, vec_id in zip(similarities[0], indices[0]):
                if vec_id == -1: continue
                try:
                    metadata = self._get_metadata_for_id(vec_id)
                    results.append({"similarity": float(sim), "metadata": metadata})
                except Exception as e:
                    logging.error(f"Could not retrieve metadata for vector ID {vec_id}: {e}")

        logging.info(f"Found {len(results)} results.")
        return results

    def get_results_from_cluster(self, cluster_id: int, top_n: int = 5) -> List[Dict]:
        if cluster_id not in self.cluster_vector_map:
            logging.error(f"Cluster ID {cluster_id} does not exist.")
            return []

        logging.info(f"Retrieving top {top_n} results from cluster {cluster_id}")
        info = self.cluster_vector_map[cluster_id]
        start_index = info['start_index']
        count = info['vector_count']

        results = []
        for i in range(min(top_n, count)):
            vector_id = start_index + i
            metadata = self._get_metadata_for_id(vector_id)
            # Fake similarity: higher for items at the start of the cluster
            results.append({"similarity": 1.0 - (i / count), "metadata": metadata})
        return results

    # --- UI and Helper methods (unchanged) ---
    def display_results(self, results: List[Dict], detailed: bool = False, max_text_length: int = 300):
        if not results:
            print("❌ No results found!")
            return
        print(f"\n🎯 Search Results ({len(results)} found):")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            similarity = result['similarity']
            text = metadata.get('text', '')
            if len(text) > max_text_length and not detailed:
                text = text[:max_text_length] + "..."

            print(f"\n📄 Result #{i} (Similarity: {similarity:.4f})")
            print(f"📚 Source: {metadata.get('source', 'Unknown')}")
            print(f"📝 Words: {metadata.get('word_count', 0)}")
            if detailed:
                print(f"🔢 Chunk ID: {metadata.get('chunk_id', 'N/A')}")
                print(f"⏰ Timestamp: {metadata.get('timestamp', 'N/A')}")
            print(f"📖 Content:\n   {text}")
            if i < len(results):
                print("-" * 60)

    def interactive_search(self):
        print("\n🤖 Interactive Vector Search")
        print("Type your query, or use commands: /cluster <id>, /clusters, /quit")
        print("-" * 50)
        while True:
            try:
                query = input("\n🔍 Search> ").strip()
                if not query: continue
                if query.lower() in ['quit', 'exit', 'q']: break

                if query.startswith('/'):
                    self._handle_command(query)
                else:
                    results = self.search(query, top_k=5)
                    self.display_results(results)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)
        print("\n👋 Goodbye!")

    def _handle_command(self, command: str):
        parts = command.split()
        cmd = parts[0].lower()
        if cmd == '/clusters':
            total = self.manifest['metadata']['total_clusters']
            print(f"📦 Found {total} clusters. Use '/cluster <id>' to view one.")
        elif cmd == '/cluster' and len(parts) > 1:
            try:
                cluster_id = int(parts[1])
                results = self.get_results_from_cluster(cluster_id, top_n=5)
                self.display_results(results)
            except (ValueError, IndexError):
                print("❌ Invalid cluster ID. Usage: /cluster <number>")
        else:
            print("❌ Unknown command. Available: /clusters, /cluster <id>, /quit")


def main():
    parser = argparse.ArgumentParser(
        description="Search a vector database created by the VectorPipelineManager.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ... (rest of main function is unchanged)
    parser.add_argument("query", nargs='?', help="Search query (optional in interactive mode)")
    parser.add_argument("--mp4", default="wikipedia_vectors.mp4", help="Path to MP4 vector database")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--detailed", action="store_true", help="Show detailed metadata")
    parser.add_argument("--interactive", action="store_true", help="Start interactive search session")
    parser.add_argument("--list-clusters", action="store_true", help="List cluster info and exit")

    args = parser.parse_args()

    try:
        search_engine = VectorSearchEngine(args.mp4, args.model)

        if args.list_clusters:
            total = search_engine.manifest['metadata']['total_clusters']
            print(f"📦 Total clusters available: {total}")
            return 0

        if args.interactive:
            search_engine.interactive_search()
            return 0

        if not args.query:
            parser.error("A search query is required in non-interactive mode.")

        results = search_engine.search(query=args.query, top_k=args.top_k)
        search_engine.display_results(results, detailed=args.detailed)

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Ensure the .mp4, _manifest.json, and _faiss.index files all exist.")
        return 1
    except Exception as e:
        logging.error("A critical error occurred.", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())