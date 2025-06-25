#!/usr/bin/env python3
"""
Wikipedia MP4 Vector Search CLI (Corrected)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class VectorSearchEngine:
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

        self.fragment_vector_map = self._build_fragment_map()
        self.cluster_cache = {}
        self._print_database_info()

    def _build_fragment_map(self) -> Dict[int, Dict]:
        logging.info("Building fragment-to-vector-index map...")
        fragment_map = {}
        sorted_fragments = sorted(
            self.manifest['metadata']['file_layout']['fragments'],
            key=lambda f: f['fragment_id']
        )
        current_vector_index = 0
        for fragment in sorted_fragments:
            fragment_id = fragment['fragment_id']
            vector_count = fragment['vector_count']
            fragment_map[fragment_id] = {
                'start_index': current_vector_index,
                'vector_count': vector_count,
                'source': fragment.get('source', 'Unknown'),
                'file_layout': fragment
            }
            current_vector_index += vector_count
        return fragment_map

    def _validate_paths(self):
        for path in [self.mp4_path, self.manifest_path, self.index_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

    def _print_database_info(self):
        meta = self.manifest["metadata"]
        print("\n📊 Database Information:")
        print("=" * 50)
        print(f"📚 Total vectors: {meta['total_vectors']:,}")
        print(f"🎯 Vector dimensions: {meta['vector_dim']}")
        print(f"📦 Fragments (Articles): {meta['total_fragments']}")
        print(f"🔍 Index type: {meta.get('faiss_index_type', 'Unknown')}")
        print("=" * 50)

    def _find_fragment_for_vector_id(self, vector_id: int) -> Tuple[int, int, str]:
        for fragment_id, info in self.fragment_vector_map.items():
            start_index = info['start_index']
            count = info['vector_count']
            if start_index <= vector_id < start_index + count:
                local_index = vector_id - start_index
                return fragment_id, local_index, info['source']
        raise ValueError(f"Vector ID {vector_id} not found.")

    def _get_full_article_content(self, fragment_id: int) -> str:
        """
        Reads all non-overlapping chunks from a fragment and joins them to
        form the full, correct article text.
        """
        # First, get the list of metadata dictionaries for the fragment
        if fragment_id in self.cluster_cache:
            metadata_list = self.cluster_cache[fragment_id]
        else:
            # This logic to read from the file is correct
            fragment_info = self.fragment_vector_map[fragment_id]['file_layout']
            with open(self.mp4_path, 'rb') as f:
                f.seek(fragment_info['mdat_offset'] + 8)
                vector_bytes_size = fragment_info['vector_count'] * self.manifest['metadata']['vector_dim'] * 4
                f.seek(vector_bytes_size, 1)
                metadata_size = struct.unpack('>I', f.read(4))[0]
                metadata_list = json.loads(f.read(metadata_size).decode('utf-8'))
                self.cluster_cache[fragment_id] = metadata_list

        # The chunks are already in order because of how we built the fragments.
        # Joining them with a space will now correctly reconstruct the article.
        full_text = " ".join([chunk['text'] for chunk in metadata_list])
        return full_text

    ### --- NEW TWO-STAGE SEARCH LOGIC --- ###
    def search_and_rerank_articles(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Performs a two-stage search with a hybrid re-ranking score
        to return the most relevant full articles.
        """
        logging.info(f"Stage 1: Performing broad search for query: '{query}'")

        broad_k = max(100, top_k * 20)
        query_vector = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vector)
        similarities, indices = self.index.search(query_vector, broad_k)

        logging.info("Stage 2: Calculating hybrid scores for candidate articles.")

        # This dictionary will now store more info for the hybrid score
        article_scores = {}
        # Example: {'wikipedia_en_AI': {'total_score': 1.5, 'best_score': 0.8, 'fragment_id': 1}}

        if indices.size > 0:
            for sim, vec_id in zip(similarities[0], indices[0]):
                if vec_id == -1: continue
                try:
                    fragment_id, _, source = self._find_fragment_for_vector_id(vec_id)

                    if source not in article_scores:
                        article_scores[source] = {
                            'total_score': 0.0,
                            'best_score': 0.0,
                            'fragment_id': fragment_id
                        }

                    # Add to the total score (the density part)
                    article_scores[source]['total_score'] += sim
                    # Update the best score if this chunk is better (the quality part)
                    if sim > article_scores[source]['best_score']:
                        article_scores[source]['best_score'] = sim

                except ValueError:
                    continue

        if not article_scores:
            return []

        # --- This is the key change: The new scoring formula ---
        # A weight to emphasize the quality of the single best match.
        BEST_MATCH_WEIGHT = 4.0

        # Calculate the final hybrid score for each article
        final_ranked_articles = []
        for source, data in article_scores.items():
            hybrid_score = data['total_score'] + (data['best_score'] * BEST_MATCH_WEIGHT)
            final_ranked_articles.append({
                'source': source,
                'score': hybrid_score,
                'fragment_id': data['fragment_id']
            })

        # Sort articles by the new hybrid score
        sorted_articles = sorted(final_ranked_articles, key=lambda x: x['score'], reverse=True)

        logging.info("Stage 3: Retrieving full content for top re-ranked articles.")
        results = []
        for article_data in sorted_articles[:top_k]:
            full_text = self._get_full_article_content(article_data['fragment_id'])
            results.append({
                "source": article_data['source'],
                "relevance_score": article_data['score'],
                "content": full_text
            })

        return results

    def display_full_article_results(self, results: List[Dict]):
        """Displays the re-ranked results, which are full articles."""
        if not results:
            print("❌ No relevant articles found.")
            return

        print(f"\n🎯 Top {len(results)} Relevant Articles:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n📄 Result #{i} (Relevance Score: {result['relevance_score']:.4f})")
            print(f"📚 Source: {result['source']}")
            print("-" * 30)
            print(f"📖 Article Content:\n{result['content']}")
            if i < len(results):
                print("\n" + "=" * 80)

    # Delete the old search() and display_results() methods as they are now replaced.
    # The interactive mode will also need to be updated to use the new methods.

    def interactive_search(self):
        print("\n🤖 Interactive Article Search")
        print("Type your query or 'quit' to exit.")
        print("-" * 50)
        while True:
            try:
                query = input("\n🔍 Search Articles> ").strip()
                if not query: continue
                if query.lower() in ['quit', 'exit', 'q']: break

                # We will simplify the interactive mode to only do article search
                results = self.search_and_rerank_articles(query, top_k=3)
                self.display_full_article_results(results)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)
        print("\n👋 Goodbye!")

def main():
    parser = argparse.ArgumentParser(
        description="Search Wikipedia articles using a two-stage re-ranking process.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs='?', help="Search query (required in non-interactive mode)")
    parser.add_argument("--mp4", default="wikipedia_vectors.mp4", help="Path to MP4 vector database")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of full articles to return")
    parser.add_argument("--interactive", action="store_true", help="Start interactive search session")

    args = parser.parse_args()

    try:
        search_engine = VectorSearchEngine(args.mp4, args.model)

        if args.interactive:
            search_engine.interactive_search()
            return 0

        if not args.query:
            parser.error("A search query is required in non-interactive mode.")

        # Call the new search and display methods
        results = search_engine.search_and_rerank_articles(query=args.query, top_k=args.top_k)
        search_engine.display_full_article_results(results)

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Ensure the .mp4, _manifest.json, and _faiss.index files all exist.")
        return 1
    except Exception as e:
        logging.error("A critical error occurred.", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())