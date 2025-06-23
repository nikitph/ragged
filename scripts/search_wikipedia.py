#!/usr/bin/env python3
"""
Wikipedia MP4 Vector Search CLI

Search through your Wikipedia MP4 vector database using natural language queries.
This script imports your existing VectorMP4Decoder class.

Usage:
    python search_wikipedia.py "machine learning algorithms"
    python search_wikipedia.py "how do neural networks work" --top-k 5
    python search_wikipedia.py "python programming" --detailed
    python search_wikipedia.py --interactive
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import sys
import os
from datetime import datetime

# Import your existing decoder
from decoder import VectorMP4Decoder


class WikipediaSearchEngine:
    """
    Search engine for Wikipedia MP4 vector databases

    Think of this as your personal Wikipedia search assistant that understands
    the meaning behind your questions, not just keywords.
    """

    def __init__(self, mp4_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the search engine

        Args:
            mp4_path: Path to the Wikipedia MP4 vector database
            model_name: Same model used for encoding (must match!)
        """
        self.mp4_path = mp4_path
        self.model_name = model_name

        # Load the same model used for encoding vectors
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Initialize your existing decoder
        print(f"🔄 Loading MP4 vector database: {mp4_path}")
        self.decoder = VectorMP4Decoder(mp4_path)

        # Print database info
        self._print_database_info()

    def _print_database_info(self):
        """Display information about the loaded database"""
        manifest = self.decoder.manifest
        metadata = manifest["metadata"]

        print("\n📊 Database Information:")
        print("=" * 50)
        print(f"📚 Total vectors: {metadata['total_vectors']:,}")
        print(f"🧩 Vector dimensions: {metadata['vector_dim']}")
        print(f"📦 Fragments: {len(metadata['fragments'])}")
        print(f"🔍 Index type: {metadata['faiss_index_type']}")

        # Get sample topics
        all_topics = set()
        for frag in metadata['fragments']:
            all_topics.update(frag.get('topics', []))

        if all_topics:
            sample_topics = list(all_topics)[:10]
            print(f"🏷️  Sample topics: {', '.join(sample_topics)}")

        print("=" * 50)

    def search(self,
               query: str,
               top_k: int = 10,
               topic: str = None,
               min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for relevant content using natural language query

        Args:
            query: Natural language search query
            top_k: Number of results to return
            topic: Optional topic filter
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of search results with content and metadata
        """
        if not query.strip():
            return []

        print(f"🔍 Searching for: '{query}'")
        if topic:
            print(f"🏷️  Filtered by topic: {topic}")

        # Convert query to vector using the same model
        query_vector = self.model.encode([query])[0]

        # Search using your existing decoder
        results = self.decoder.search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            topic=topic
        )

        # Filter by minimum similarity if specified
        if min_similarity > 0.0:
            results = [r for r in results if r['similarity'] >= min_similarity]

        print(f"✅ Found {len(results)} results")
        return results

    def search_by_topic(self, topic: str) -> List[Dict]:
        """
        Get all content from a specific topic using your decoder

        Args:
            topic: Topic name to search for

        Returns:
            List of all vectors and metadata for that topic
        """
        print(f"🏷️  Getting all content for topic: '{topic}'")

        vectors, metadata = self.decoder.get_vectors_by_topic(topic)

        if len(vectors) == 0:
            print(f"❌ No content found for topic: {topic}")
            return []

        results = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            results.append({
                'vector': vector,
                'metadata': meta,
                'similarity': 1.0  # Perfect match for topic search
            })

        print(f"✅ Found {len(results)} results for topic '{topic}'")
        return results

    def get_available_topics(self) -> List[str]:
        """Get all available topics in the database"""
        all_topics = set()

        for frag in self.decoder.manifest["metadata"]["fragments"]:
            all_topics.update(frag.get("topics", []))

        return sorted(list(all_topics))

    def display_results(self,
                        results: List[Dict],
                        detailed: bool = False,
                        max_text_length: int = 300) -> None:
        """
        Display search results in a formatted way

        Args:
            results: Search results from search() method
            detailed: Show full metadata and longer text
            max_text_length: Maximum characters to show for each result
        """
        if not results:
            print("❌ No results found!")
            return

        print(f"\n🎯 Search Results ({len(results)} found):")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            similarity = result['similarity']

            # Get the main text content
            text = metadata.get('text', '')
            if len(text) > max_text_length and not detailed:
                text = text[:max_text_length] + "..."

            # Extract key information
            source = metadata.get('source', 'Unknown source')
            topic = metadata.get('topic', 'No topic')
            word_count = metadata.get('word_count', 0)

            # Format the result
            print(f"\n📄 Result #{i} (Similarity: {similarity:.3f})")
            print(f"📚 Source: {source}")
            print(f"🏷️  Topic: {topic}")
            print(f"📝 Words: {word_count}")

            if detailed:
                chunk_id = metadata.get('chunk_id', 'N/A')
                timestamp = metadata.get('timestamp', 'N/A')
                text_hash = metadata.get('text_hash', 'N/A')
                print(f"🔢 Chunk ID: {chunk_id}")
                print(f"⏰ Processed: {timestamp}")
                print(f"🔗 Hash: {text_hash}")

            print(f"📖 Content:")
            print(f"   {text}")

            if i < len(results):
                print("-" * 60)

    def interactive_search(self):
        """
        Start an interactive search session

        This is like having a conversation with your Wikipedia database!
        """
        print("\n🤖 Interactive Wikipedia Search")
        print("Type your questions or 'quit' to exit")
        print("Commands:")
        print("  /topics - List all available topics")
        print("  /topic <name> - Search by specific topic")
        print("  /help - Show this help")
        print("-" * 50)

        while True:
            try:
                query = input("\n🔍 Search> ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if query.startswith('/'):
                    self._handle_command(query)
                    continue

                # Regular search
                results = self.search(query, top_k=5)
                self.display_results(results, detailed=False)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _handle_command(self, command: str):
        """Handle special commands in interactive mode"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == '/topics':
            topics = self.get_available_topics()
            print(f"\n🏷️  Available topics ({len(topics)}):")
            for i, topic in enumerate(topics[:20], 1):  # Show first 20
                print(f"  {i:2d}. {topic}")
            if len(topics) > 20:
                print(f"  ... and {len(topics) - 20} more")

        elif cmd == '/topic' and len(parts) > 1:
            topic = ' '.join(parts[1:])
            results = self.search_by_topic(topic)
            self.display_results(results[:5], detailed=False)

        elif cmd == '/help':
            print("\n🤖 Available commands:")
            print("  /topics - List all available topics")
            print("  /topic <name> - Search by specific topic")
            print("  /help - Show this help")
            print("  quit - Exit the search")

        else:
            print("❌ Unknown command. Type /help for available commands.")


def main():
    """Command line interface for Wikipedia search"""
    parser = argparse.ArgumentParser(
        description="Search Wikipedia MP4 vector database with natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python search_wikipedia.py "machine learning algorithms"

  # Get more results
  python search_wikipedia.py "neural networks" --top-k 10

  # Filter by topic
  python search_wikipedia.py "programming" --topic "computer"

  # Detailed results
  python search_wikipedia.py "artificial intelligence" --detailed

  # Interactive mode (recommended!)
  python search_wikipedia.py --interactive

  # List all topics
  python search_wikipedia.py --list-topics
        """
    )

    parser.add_argument("query", nargs='?', help="Search query (optional in interactive mode)")
    parser.add_argument("--mp4", default="wikipedia_vectors.mp4",
                        help="Path to MP4 vector database (default: wikipedia_vectors.mp4)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Embedding model name (must match encoding model)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to return (default: 10)")
    parser.add_argument("--topic", help="Filter results by topic")
    parser.add_argument("--min-similarity", type=float, default=0.0,
                        help="Minimum similarity threshold (0.0-1.0)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed metadata for each result")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive search session")
    parser.add_argument("--list-topics", action="store_true",
                        help="List all available topics and exit")
    parser.add_argument("--max-text", type=int, default=300,
                        help="Maximum text length to display (default: 300)")

    args = parser.parse_args()

    # Check if MP4 file exists
    if not os.path.exists(args.mp4):
        print(f"❌ MP4 file not found: {args.mp4}")
        print("Make sure you've run the Wikipedia encoder first!")
        return 1

    try:
        # Initialize search engine
        search_engine = WikipediaSearchEngine(args.mp4, args.model)

        # Handle different modes
        if args.list_topics:
            topics = search_engine.get_available_topics()
            print(f"\n🏷️  Available topics ({len(topics)}):")
            for i, topic in enumerate(topics, 1):
                print(f"  {i:3d}. {topic}")
            return 0

        if args.interactive:
            search_engine.interactive_search()
            return 0

        if not args.query:
            print("❌ Please provide a search query or use --interactive mode")
            parser.print_help()
            return 1

        # Perform search
        results = search_engine.search(
            query=args.query,
            top_k=args.top_k,
            topic=args.topic,
            min_similarity=args.min_similarity
        )

        # Display results
        search_engine.display_results(
            results,
            detailed=args.detailed,
            max_text_length=args.max_text
        )

        return 0

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure all required files exist:")
        print(f"  - {args.mp4}")
        print(f"  - {args.mp4.replace('.mp4', '_manifest.json')}")
        print(f"  - {args.mp4.replace('.mp4', '_faiss.index')}")
        return 1
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure vectorMP4Decoder.py is in the same directory!")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


# Quick demo function for testing
def demo_search():
    """
    Quick demo function to test the search functionality
    """
    print("🧪 Demo Search Session")

    try:
        # Try to find a demo MP4 file
        mp4_files = [f for f in os.listdir('../ragged/video') if f.endswith('.mp4')]
        if not mp4_files:
            print("❌ No MP4 files found in current directory")
            print("Run the Wikipedia encoder first!")
            return

        mp4_path = mp4_files[0]
        print(f"📁 Using MP4 file: {mp4_path}")

        # Initialize search engine
        search_engine = WikipediaSearchEngine(mp4_path)

        # Demo queries
        demo_queries = [
            "artificial intelligence",
            "machine learning algorithms",
            "neural networks",
            "computer programming"
        ]

        for query in demo_queries:
            print(f"\n{'=' * 60}")
            print(f"🔍 Demo Query: '{query}'")
            print('=' * 60)

            results = search_engine.search(query, top_k=3)
            search_engine.display_results(results, max_text_length=200)

            input("\nPress Enter for next query...")

        print("\n🎉 Demo completed!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


"""
USAGE INSTRUCTIONS:

1. Make sure you have these files in the same directory:
   - search_wikipedia.py (this file)
   - vectorMP4Decoder.py (your decoder class)
   - wikipedia_vectors.mp4 (or your MP4 database)
   - wikipedia_vectors_manifest.json
   - wikipedia_vectors_faiss.index

2. Install dependencies:
   pip install sentence-transformers faiss-cpu numpy

3. Run searches:

   # Simple search
   python search_wikipedia.py "machine learning"

   # Get more results  
   python search_wikipedia.py "neural networks" --top-k 15

   # Interactive mode (best experience!)
   python search_wikipedia.py --interactive

   # List all available topics
   python search_wikipedia.py --list-topics

4. Interactive mode example:
   🔍 Search> machine learning algorithms
   🔍 Search> /topics
   🔍 Search> /topic neural
   🔍 Search> how do computers learn
   🔍 Search> quit

The search understands meaning, not just keywords!
"""