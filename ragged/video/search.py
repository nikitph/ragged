#!/usr/bin/env python3
"""
Wikipedia Vector Search with Cloudflare R2 Support

Search through Wikipedia MP4 vector database using natural language queries.
Optimized for Cloudflare R2 storage with intelligent caching and range requests.

Usage:
    # Basic search
    python wikipedia_search.py "machine learning algorithms"

    # Interactive mode (recommended)
    python wikipedia_search.py --interactive

    # With performance monitoring
    python wikipedia_search.py "neural networks" --show-performance --detailed

    # Custom cache settings
    python wikipedia_search.py "AI ethics" --cache-dir ./my_cache --cache-size 200
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import sys
import os
import tempfile
from datetime import datetime
from urllib.parse import urlparse
import time

from ragged.video.model_server import ModelClient

# Default R2 URLs - you can override these with command line arguments
DEFAULT_R2_BASE = "https://pub-e12d369657534f328cc36a7331ff7bff.r2.dev"
DEFAULT_MP4_URL = f"{DEFAULT_R2_BASE}/wikipedia_vectors.mp4"
DEFAULT_MANIFEST_URL = f"{DEFAULT_R2_BASE}/wikipedia_vectors_manifest.json"
DEFAULT_FAISS_URL = f"{DEFAULT_R2_BASE}/wikipedia_vectors_faiss.index"

# Import the enhanced decoder
try:
    from CDNDecoder import CDNVectorMP4Decoder
except ImportError:
    print("❌ Error: Could not import CDNVectorMP4Decoder")
    print("Make sure 'decoder.py' with CDNVectorMP4Decoder is in the same directory!")
    print("You can download it from the artifacts provided earlier.")
    sys.exit(1)


class WikipediaSearchEngine:
    """
    Wikipedia search engine optimized for R2 storage

    Like having a personal research assistant that understands context and meaning,
    not just keywords!
    """

    def __init__(self,
                 mp4_path: str = DEFAULT_MP4_URL,
                 manifest_path: str = DEFAULT_MANIFEST_URL,
                 faiss_path: str = DEFAULT_FAISS_URL,
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: Optional[str] = None,
                 cache_size: int = 100,
                 enable_prefetching: bool = True,
                 max_retries: int = 3,
                 use_model_server: bool = True,
                 timeout: int = 60):
        """
        Initialize Wikipedia search engine

        Args:
            mp4_path: URL to Wikipedia MP4 vector database
            manifest_path: URL to manifest file
            faiss_path: URL to Faiss index file
            model_name: Embedding model (must match encoding model!)
            cache_dir: Directory for persistent caching
            cache_size: Number of fragments to keep in memory
            enable_prefetching: Whether to prefetch adjacent fragments
            max_retries: Number of retry attempts for downloads
            timeout: Request timeout in seconds
            use_model_server: Whether to use persistent model server
            model_server_host: Model server host (default: localhost)
            model_server_port: Model server port (default: 8888)
        """
        self.mp4_path = mp4_path
        self.manifest_path = manifest_path
        self.faiss_path = faiss_path
        self.model_name = model_name
        self.use_model_server = use_model_server

        # Use a persistent cache directory by default
        if cache_dir is None:
            # Create cache in user's home directory for persistence across runs
            home_cache = os.path.expanduser("~/.wikipedia_search_cache")
            self.cache_dir = home_cache
        else:
            self.cache_dir = cache_dir
        self._preload_fragments = 0  # Will be set later if needed

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"🚀 Initializing Wikipedia Search Engine...")
        print(f"📍 R2 Base: {self._get_base_url(mp4_path)}")

        # Load embedding model with local caching
        print(f"🔄 Loading embedding model: {model_name}")
        start_time = time.time()

        # Pre-download model if not already cached
        try:
            if use_model_server:
                self.client = ModelClient()
                if not self.client.is_server_running():
                    raise RuntimeError("Model server not running. Start with: python model_server.py --start")
                print("✅ Using model server - instant encoding!")
            else:
                # Load model normally (4+ seconds)
                self.model = SentenceTransformer(model_name, cache_folder=os.path.expanduser("~/.sentence_transformers"))

        except Exception as e:
            print(f"   Downloading model {model_name} (one-time setup)...")
            self.model = SentenceTransformer(model_name)

        model_time = time.time() - start_time
        print(f"✅ Model loaded in {model_time:.2f}s")

        # Initialize enhanced decoder
        print(f"🔄 Connecting to R2 vector database...")
        start_time = time.time()

        self.decoder = CDNVectorMP4Decoder(
            mp4_path=mp4_path,
            manifest_path=manifest_path,
            faiss_path=faiss_path,
            cache_size=cache_size,
            disk_cache_dir=self.cache_dir,
            enable_prefetching=enable_prefetching,
            max_retries=max_retries,
            timeout=timeout
        )

        decoder_time = time.time() - start_time
        print(f"✅ Connected to R2 in {decoder_time:.2f}s")

        # Display database info
        self._print_database_info()

    def _get_base_url(self, url: str) -> str:
        """Extract base URL for display"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _print_database_info(self):
        """Display information about the loaded database"""
        try:
            info = self.decoder.get_manifest_info()

            print("\n📊 Wikipedia Database Information:")
            print("=" * 60)
            print(f"📚 Total vectors: {info['total_vectors']:,}")
            print(f"🧩 Vector dimensions: {info['vector_dim']}")
            print(f"📦 Fragments: {info['num_fragments']}")
            print(f"💾 Cache directory: {self.cache_dir}")
            print(f"🧠 Memory cache size: {self.decoder.cache_size} fragments")
            print(f"🚀 Prefetching: {'Enabled' if self.decoder.enable_prefetching else 'Disabled'}")

            # Check storage type
            storage_info = info.get('storage_type', 'CDN')
            if 'r2.dev' in self.mp4_path or 'r2.cloudflarestorage.com' in self.mp4_path:
                storage_info = "Cloudflare R2"
            print(f"🌐 Storage: {storage_info}")

            print("=" * 60)

        except Exception as e:
            print(f"❌ Error getting database info: {e}")

    def _preload_random_fragments(self, num_fragments: int):
        """Preload random fragments to warm up the cache"""
        if not self.decoder.manifest:
            return

        import random

        available_fragments = [f["id"] for f in self.decoder.manifest["metadata"]["fragments"]]
        fragments_to_load = random.sample(available_fragments, min(num_fragments, len(available_fragments)))

        print(f"🔥 Preloading {len(fragments_to_load)} fragments for faster searches...")
        start_time = time.time()

        # Load fragments in parallel
        futures = []
        for frag_id in fragments_to_load:
            future = self.decoder.executor.submit(self.decoder._get_fragment, frag_id)
            futures.append(future)

        # Wait for completion
        completed = 0
        for future in futures:
            try:
                future.result(timeout=30)  # 30 second timeout per fragment
                completed += 1
            except Exception as e:
                print(f"Warning: Failed to preload fragment: {e}")

        preload_time = time.time() - start_time
        print(f"✅ Preloaded {completed} fragments in {preload_time:.2f}s")

    def search(self,
               query: str,
               top_k: int = 10,
               topic: str = None,
               min_similarity: float = 0.0,
               show_performance: bool = False) -> List[Dict]:
        """
        Search Wikipedia using natural language query

        Args:
            query: Natural language search query
            top_k: Number of results to return
            topic: Optional topic filter
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            show_performance: Whether to show performance metrics

        Returns:
            List of search results with content and metadata
        """
        if not query.strip():
            return []

        print(f"🔍 Searching Wikipedia for: '{query}'")
        if topic:
            print(f"🏷️  Filtered by topic: {topic}")
        if min_similarity > 0:
            print(f"📊 Minimum similarity: {min_similarity:.2f}")

        # Performance tracking
        start_time = time.time()

        # Convert query to vector (using server or local model)
        encode_start = time.time()
        if self.use_model_server:
            query_vector, encode_time_server = self.client.encode(query)
            encode_time = time.time() - encode_start
            if show_performance:
                print(f"   ⚡ Server encoding: {encode_time_server:.4f}s (total: {encode_time:.3f}s)")
        else:
            query_vector = self.model.encode([query])[0]
            encode_time = time.time() - encode_start

        # Search using decoder
        search_start = time.time()
        results = self.decoder.search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            topic=topic
        )
        search_time = time.time() - search_start

        # Filter by similarity if specified
        if min_similarity > 0.0:
            results = [r for r in results if r['similarity'] >= min_similarity]

        total_time = time.time() - start_time

        print(f"✅ Found {len(results)} results in {total_time:.3f}s")

        if show_performance and not self.use_model_server:
            print(f"   📊 Query encoding: {encode_time:.3f}s")
            print(f"   🔍 Vector search: {search_time:.3f}s")

            # Show cache info
            if hasattr(self.decoder, 'memory_cache'):
                cache_size = len(self.decoder.memory_cache)
                print(f"   💾 Fragments cached: {cache_size}")

        return results

    def search_by_topic(self, topic: str) -> List[Dict]:
        """Get all content from a specific topic"""
        print(f"🏷️  Getting all content for topic: '{topic}'")
        start_time = time.time()

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

        search_time = time.time() - start_time
        print(f"✅ Found {len(results)} results for topic '{topic}' in {search_time:.3f}s")
        return results

    def get_available_topics(self) -> List[str]:
        """Get all available topics in the database"""
        if not hasattr(self.decoder, 'manifest') or not self.decoder.manifest:
            return []

        all_topics = set()
        for frag in self.decoder.manifest["metadata"]["fragments"]:
            all_topics.update(frag.get("topics", []))

        return sorted(list(all_topics))

    def display_results(self,
                        results: List[Dict],
                        detailed: bool = False,
                        max_text_length: int = 300,
                        show_vectors: bool = False) -> None:
        """Display search results in a formatted way"""
        if not results:
            print("❌ No results found!")
            return

        print(f"\n🎯 Wikipedia Search Results ({len(results)} found):")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            similarity = result['similarity']

            # Get text content
            text = metadata.get('text', 'No content available')
            if len(text) > max_text_length and not detailed:
                text = text[:max_text_length] + "..."

            # Extract key information
            source = metadata.get('source', 'Unknown source')
            topic = metadata.get('topic', 'No topic')
            word_count = metadata.get('word_count', 0)

            # Format with color-coded similarity
            if similarity > 0.8:
                sim_emoji = "🟢"
            elif similarity > 0.6:
                sim_emoji = "🟡"
            else:
                sim_emoji = "🔴"

            print(f"\n📄 Result #{i} {sim_emoji} (Similarity: {similarity:.3f})")
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

            if show_vectors and 'vector' in result:
                vector = result['vector']
                print(f"🧮 Vector: shape={vector.shape}, norm={np.linalg.norm(vector):.3f}")

            print(f"📖 Content:")
            print(f"   {text}")

            if i < len(results):
                print("-" * 60)

    def interactive_search(self):
        """Start interactive search session"""
        print("\n🤖 Interactive Wikipedia Search")
        print("Ask any question about topics in Wikipedia!")
        print("\nCommands:")
        print("  /topics - List available topics")
        print("  /topic <name> - Search by specific topic")
        print("  /cache - Show cache statistics")
        print("  /info - Show database information")
        print("  /help - Show this help")
        print("  quit - Exit")
        print("-" * 60)

        while True:
            try:
                query = input("\n🔍 Ask Wikipedia> ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using Wikipedia Search!")
                    break

                if query.startswith('/'):
                    self._handle_command(query)
                    continue

                # Regular search
                start_time = time.time()
                results = self.search(query, top_k=5)

                if results:
                    # In interactive mode, show condensed results
                    print(f"\n🎯 Top {min(3, len(results))} Results:")
                    for i, result in enumerate(results[:3], 1):
                        metadata = result['metadata']
                        similarity = result['similarity']
                        text = metadata.get('text', '')[:200] + "..."

                        sim_emoji = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.5 else "🔴"
                        print(f"\n{i}. {sim_emoji} {metadata.get('source', 'Unknown')} (sim: {similarity:.3f})")
                        print(f"   {text}")

                    if len(results) > 3:
                        print(f"\n   ... and {len(results) - 3} more results")

                total_time = time.time() - start_time
                print(f"\n⏱️  Search completed in {total_time:.3f}s")

            except KeyboardInterrupt:
                print("\n👋 Thanks for using Wikipedia Search!")
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
            # Show topics in columns for better readability
            for i in range(0, len(topics[:20]), 4):
                row_topics = topics[i:i + 4]
                print("   " + "  ".join(f"{j + i + 1:2d}. {topic:<20}" for j, topic in enumerate(row_topics)))
            if len(topics) > 20:
                print(f"   ... and {len(topics) - 20} more topics")

        elif cmd == '/topic' and len(parts) > 1:
            topic = ' '.join(parts[1:])
            results = self.search_by_topic(topic)
            if results:
                print(f"\n🎯 Content for topic '{topic}':")
                for i, result in enumerate(results[:3], 1):
                    metadata = result['metadata']
                    text = metadata.get('text', '')[:150] + "..."
                    print(f"   {i}. {metadata.get('source', 'Unknown')}")
                    print(f"      {text}")

        elif cmd == '/cache':
            print(f"\n💾 Cache Statistics:")
            print(f"   Memory cache: {len(self.decoder.memory_cache)} fragments")
            print(f"   Cache directory: {self.cache_dir}")

            # Count disk cache files
            if os.path.exists(self.cache_dir):
                cache_files = [f for f in os.listdir(self.cache_dir)
                               if f.endswith('.pkl') or f.endswith('.json') or f.endswith('.index')]
                print(f"   Disk cache files: {len(cache_files)}")

        elif cmd == '/info':
            self._print_database_info()

        elif cmd == '/help':
            print("\n🤖 Available commands:")
            print("  /topics - List all available topics")
            print("  /topic <name> - Search by specific topic")
            print("  /cache - Show cache statistics")
            print("  /info - Show database information")
            print("  /help - Show this help")
            print("  quit - Exit the search")

        else:
            print("❌ Unknown command. Type /help for available commands.")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.decoder, 'cleanup'):
            self.decoder.cleanup()
        print(f"🧹 Cleanup completed. Cache saved in: {self.cache_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Command line interface for Wikipedia search"""
    parser = argparse.ArgumentParser(
        description="Search Wikipedia vector database with natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic search with default R2 URLs
  python wikipedia_search.py "machine learning algorithms"

  # Interactive mode (recommended!)
  python wikipedia_search.py --interactive

  # Custom URLs
  python wikipedia_search.py "neural networks" \\
    --mp4 "https://your-bucket.r2.dev/wikipedia.mp4" \\
    --manifest "https://your-bucket.r2.dev/wikipedia_manifest.json" \\
    --faiss "https://your-bucket.r2.dev/wikipedia_faiss.index"

  # Model server usage (ultra-fast!)
  python wikipedia_search.py "machine learning" --use-server

  # Performance monitoring with server
  python wikipedia_search.py "artificial intelligence" --use-server --show-performance

  # Custom server location
  python wikipedia_search.py "neural networks" --use-server --server-host 192.168.1.100 --server-port 9999

  # Custom cache settings
  python wikipedia_search.py "quantum computing" --cache-dir ./my_cache --cache-size 200

Default R2 URLs:
  MP4: {DEFAULT_MP4_URL}
  Manifest: {DEFAULT_MANIFEST_URL}
  Faiss: {DEFAULT_FAISS_URL}
        """
    )

    # Query argument
    parser.add_argument("query", nargs='?',
                        help="Search query (optional in interactive mode)")

    # File location arguments (with your R2 URLs as defaults)
    parser.add_argument("--mp4", default=DEFAULT_MP4_URL,
                        help=f"URL to MP4 vector database (default: your R2 URL)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_URL,
                        help=f"URL to manifest file (default: your R2 URL)")
    parser.add_argument("--faiss", default=DEFAULT_FAISS_URL,
                        help=f"URL to Faiss index file (default: your R2 URL)")

    # Model and search parameters
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Embedding model name (must match encoding model)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to return (default: 10)")
    parser.add_argument("--topic",
                        help="Filter results by topic")
    parser.add_argument("--min-similarity", type=float, default=0.0,
                        help="Minimum similarity threshold (0.0-1.0)")

    # Performance and caching
    parser.add_argument("--cache-dir",
                        help="Directory for persistent cache (default: temp dir)")
    parser.add_argument("--cache-size", type=int, default=100,
                        help="Memory cache size in fragments (default: 100)")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Disable fragment prefetching")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum retry attempts for downloads")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Request timeout in seconds")

    # Display options
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed metadata for each result")
    parser.add_argument("--show-vectors", action="store_true",
                        help="Show vector information in results")
    parser.add_argument("--show-performance", action="store_true",
                        help="Show performance metrics")
    parser.add_argument("--max-text", type=int, default=300,
                        help="Maximum text length to display (default: 300)")

    # Model server options
    parser.add_argument("--use-server", action="store_true",
                        help="Use persistent model server for instant encoding")
    parser.add_argument("--server-host", default="localhost",
                        help="Model server host (default: localhost)")
    parser.add_argument("--server-port", type=int, default=8888,
                        help="Model server port (default: 8888)")

    # Mode options
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive search session")
    parser.add_argument("--list-topics", action="store_true",
                        help="List all available topics and exit")
    parser.add_argument("--test-connection", action="store_true",
                        help="Test connection to R2 files and exit")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear all cached files and exit")

    args = parser.parse_args()

    # Test connection if requested
    if args.test_connection:
        test_r2_connection(args.mp4, args.manifest, args.faiss)
        return 0

    # Clear cache if requested
    if args.clear_cache:
        clear_cache(args.cache_dir)
        return 0

    try:
        # Initialize search engine
        print("🌐 Wikipedia Vector Search with Cloudflare R2")

        with WikipediaSearchEngine(
                mp4_path=args.mp4,
                manifest_path=args.manifest,
                faiss_path=args.faiss,
                model_name=args.model,
                cache_dir=args.cache_dir,
                cache_size=args.cache_size,
                enable_prefetching=not args.no_prefetch,
                max_retries=args.max_retries,
                timeout=args.timeout
        ) as search_engine:

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
                print("Example: python wikipedia_search.py \"machine learning\"")
                return 1

            # Perform search
            print(f"\n🎯 Searching Wikipedia...")
            results = search_engine.search(
                query=args.query,
                top_k=args.top_k,
                topic=args.topic,
                min_similarity=args.min_similarity,
                show_performance=args.show_performance
            )

            # Display results
            search_engine.display_results(
                results,
                detailed=args.detailed,
                max_text_length=args.max_text,
                show_vectors=args.show_vectors
            )

            return 0

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies:")
        print("  pip install sentence-transformers faiss-cpu numpy requests")
        print("And that decoder.py is in the same directory!")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.show_performance:
            import traceback
            traceback.print_exc()
        return 1


def warm_model_cache(model_name="all-MiniLM-L6-v2"):
    """Pre-download and cache the embedding model"""
    print(f"🔥 Warming model cache for: {model_name}")
    print("This is a one-time setup that will speed up future searches...")

    try:
        start_time = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time

        # Test the model with a sample encoding
        test_vector = model.encode(["test sentence"])

        print(f"✅ Model cached successfully in {load_time:.2f}s")
        print(f"📏 Vector dimension: {test_vector.shape[1]}")
        print(f"📁 Model cache location: {model.cache_folder}")
        print("🚀 Future searches will be much faster!")

    except Exception as e:
        print(f"❌ Error warming model cache: {e}")


def clear_cache(cache_dir=None):
    """Clear all cached files"""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.wikipedia_search_cache")

    if not os.path.exists(cache_dir):
        print("🧹 No cache directory found - nothing to clear")
        return

    print(f"🧹 Clearing cache directory: {cache_dir}")

    try:
        import shutil
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")


def check_model_server(host="localhost", port=8888):
    """Check if model server is running"""
    try:
        from model_server import ModelClient
        client = ModelClient(host, port)

        if client.is_server_running():
            # Test with a quick encode
            vector, encode_time = client.encode("test")
            print(f"✅ Model server is running at {host}:{port}")
            print(f"   Response time: {encode_time:.4f}s")
            print(f"   Vector dimension: {len(vector)}")
            return True
        else:
            print(f"❌ Model server not responding at {host}:{port}")
            return False
    except ImportError:
        print("❌ model_server.py not found - cannot check server status")
        return False
    except Exception as e:
        print(f"❌ Error checking model server: {e}")
        return False


def test_r2_connection(mp4_url, manifest_url, faiss_url):
    """Test connection to R2 files"""
    import requests

    print("🔧 Testing R2 Connection...")

    files_to_test = [
        ("Manifest", manifest_url),
        ("Faiss Index", faiss_url),
        ("MP4 File", mp4_url)
    ]

    all_good = True
    for name, url in files_to_test:
        try:
            print(f"   Testing {name}...", end=" ")
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                size_mb = int(response.headers.get('content-length', 0)) / 1024 / 1024
                print(f"✅ OK ({size_mb:.1f} MB)")
            else:
                print(f"❌ HTTP {response.status_code}")
                all_good = False
        except Exception as e:
            print(f"❌ Error: {e}")
            all_good = False

    if all_good:
        print("🎯 All R2 files are accessible! You're ready to search.")
    else:
        print("⚠️  Some files are not accessible. Check your R2 URLs and permissions.")


# Quick demo functions
def demo_basic_search():
    """Quick demo of basic search functionality"""
    print("🧪 Demo: Basic Wikipedia Search")

    try:
        with WikipediaSearchEngine() as engine:
            queries = ["machine learning", "quantum physics", "artificial intelligence"]

            for query in queries:
                print(f"\n--- Searching: {query} ---")
                results = engine.search(query, top_k=2)

                if results:
                    for i, result in enumerate(results, 1):
                        metadata = result['metadata']
                        text = metadata.get('text', '')[:100] + "..."
                        print(f"{i}. {metadata.get('source', 'Unknown')} (sim: {result['similarity']:.3f})")
                        print(f"   {text}")
                else:
                    print("   No results found")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    sys.exit(main())

# Uncomment to run quick demo
# demo_basic_search()