#!/usr/bin/env python3
"""
Wikipedia Vector Search with Knowledge Graph Support and Cloudflare R2

Search through Wikipedia MP4 vector database using natural language queries.
Now with knowledge graph support for deep inference and entity relationships.
Optimized for Cloudflare R2 storage with intelligent caching and range requests.

Usage:
    # Basic vector search
    python wikipedia_search.py "machine learning algorithms"

    # Hybrid search (vector + knowledge graph)
    python wikipedia_search.py "machine learning" --entity "OpenAI" --relation "founded"

    # Entity-focused search
    python wikipedia_search.py --search-entity "Elon Musk" --entity-type "PERSON"

    # Interactive mode (recommended)
    python wikipedia_search.py --interactive

    # Knowledge graph exploration
    python wikipedia_search.py --kg-stats
    python wikipedia_search.py --entity-neighborhood "Tesla" --depth 2

    # With performance monitoring
    python wikipedia_search.py "neural networks" --show-performance --detailed
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
import networkx as nx

from ragged.video.model_server import ModelClient

# Default R2 URLs - you can override these with command line arguments
DEFAULT_R2_BASE = "https://pub-e12d369657534f328cc36a7331ff7bff.r2.dev"
DEFAULT_MP4_URL = f"{DEFAULT_R2_BASE}/wikipedia_vectors.mp4"
DEFAULT_MANIFEST_URL = f"{DEFAULT_R2_BASE}/wikipedia_vectors_manifest.json"
DEFAULT_FAISS_URL = f"{DEFAULT_R2_BASE}/wikipedia_vectors_faiss.index"

# Import the enhanced decoder with KG support
try:
    from kg_enhanced_decoder import CDNVectorMP4Decoder, KGSearchResult
except ImportError:
    print("❌ Error: Could not import CDNVectorMP4Decoder with KG support")
    print("Make sure the enhanced decoder with knowledge graph support is available!")
    print("You can download it from the artifacts provided earlier.")
    sys.exit(1)


class WikipediaSearchEngine:
    """
    Wikipedia search engine with Knowledge Graph support, optimized for R2 storage

    Like having a personal research assistant that understands both context/meaning
    AND entity relationships - not just keywords!
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
                 timeout: int = 60,
                 enable_kg: bool = True):
        """
        Initialize Wikipedia search engine with Knowledge Graph support

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
            enable_kg: Whether to enable knowledge graph functionality
        """
        self.mp4_path = mp4_path
        self.manifest_path = manifest_path
        self.faiss_path = faiss_path
        self.model_name = model_name
        self.use_model_server = use_model_server
        self.enable_kg = enable_kg

        # Use a persistent cache directory by default
        if cache_dir is None:
            # Create cache in user's home directory for persistence across runs
            home_cache = os.path.expanduser("~/.wikipedia_search_cache")
            self.cache_dir = home_cache
        else:
            self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"🚀 Initializing Wikipedia Search Engine with Knowledge Graph...")
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
                self.model = SentenceTransformer(model_name,
                                                 cache_folder=os.path.expanduser("~/.sentence_transformers"))

        except Exception as e:
            print(f"   Downloading model {model_name} (one-time setup)...")
            self.model = SentenceTransformer(model_name)

        model_time = time.time() - start_time
        print(f"✅ Model loaded in {model_time:.2f}s")

        # Initialize enhanced decoder with KG support
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
            timeout=timeout,
            enable_kg=enable_kg
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
            print("=" * 70)
            print(f"📚 Total vectors: {info['total_vectors']:,}")
            print(f"🧩 Vector dimensions: {info['vector_dim']}")
            print(f"📦 Vector fragments: {info['num_fragments']}")

            # Knowledge Graph info
            if 'knowledge_graph' in info and info['knowledge_graph']['enabled']:
                kg_info = info['knowledge_graph']
                print(f"🔗 Knowledge Graph: Enabled")
                print(f"   👤 Entities: {kg_info['total_entities']:,}")
                print(f"   🔗 Relations: {kg_info['total_relations']:,}")
                print(f"   📦 KG fragments: {kg_info['num_kg_fragments']}")
                print(f"   🏷️  Entity types: {len(kg_info.get('entity_types', {}))}")
                print(f"   ↔️  Relation types: {len(kg_info.get('relation_types', {}))}")
            else:
                print(f"🔗 Knowledge Graph: Not available")

            print(f"💾 Cache directory: {self.cache_dir}")
            print(f"🧠 Memory cache size: {self.decoder.cache_size} fragments")
            print(f"🚀 Prefetching: {'Enabled' if self.decoder.enable_prefetching else 'Disabled'}")

            # Check storage type
            storage_info = "CDN"
            if 'r2.dev' in self.mp4_path or 'r2.cloudflarestorage.com' in self.mp4_path:
                storage_info = "Cloudflare R2"
            print(f"🌐 Storage: {storage_info}")

            print("=" * 70)

        except Exception as e:
            print(f"❌ Error getting database info: {e}")

    def search(self,
               query: str,
               top_k: int = 10,
               topic: str = None,
               min_similarity: float = 0.0,
               show_performance: bool = False) -> List[Dict]:
        """
        Search Wikipedia using natural language query (vector search only)

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

        # Convert query to vector
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

        return results

    def hybrid_search(self,
                      query: str,
                      entity_filter: Optional[str] = None,
                      relation_filter: Optional[str] = None,
                      top_k: int = 10,
                      show_performance: bool = False) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and knowledge graph filtering

        Args:
            query: Natural language search query
            entity_filter: Filter by entity text (e.g., "Tesla", "OpenAI")
            relation_filter: Filter by relation type (e.g., "founded", "works_at")
            top_k: Number of results to return
            show_performance: Whether to show performance metrics

        Returns:
            List of search results with KG context
        """
        if not self.enable_kg:
            print("⚠️  Knowledge graph not enabled - falling back to vector search")
            return self.search(query, top_k=top_k, show_performance=show_performance)

        print(f"🔍🔗 Hybrid search for: '{query}'")
        if entity_filter:
            print(f"   👤 Entity filter: {entity_filter}")
        if relation_filter:
            print(f"   🔗 Relation filter: {relation_filter}")

        start_time = time.time()

        # Convert query to vector
        if self.use_model_server:
            query_vector, _ = self.client.encode(query)
        else:
            query_vector = self.model.encode([query])[0]

        # Perform hybrid search
        results = self.decoder.hybrid_search(
            query_vector=query_vector,
            entity_filter=entity_filter,
            relation_filter=relation_filter,
            top_k=top_k
        )

        total_time = time.time() - start_time
        print(f"✅ Found {len(results)} hybrid results in {total_time:.3f}s")

        return results

    def search_entities(self,
                        entity_text: str,
                        entity_type: Optional[str] = None,
                        show_subgraph: bool = False) -> List[KGSearchResult]:
        """
        Search for entities in the knowledge graph

        Args:
            entity_text: Entity text to search for
            entity_type: Optional entity type filter (PERSON, ORG, GPE, etc.)
            show_subgraph: Whether to display subgraph information

        Returns:
            List of KG search results
        """
        if not self.enable_kg:
            print("❌ Knowledge graph not enabled")
            return []

        print(f"🔍👤 Searching entities for: '{entity_text}'")
        if entity_type:
            print(f"   🏷️  Entity type: {entity_type}")

        start_time = time.time()
        results = self.decoder.search_entities(entity_text, entity_type)
        search_time = time.time() - start_time

        print(f"✅ Found {len(results)} entity results in {search_time:.3f}s")

        if show_subgraph and results:
            for i, result in enumerate(results):
                print(f"\nSubgraph {i + 1}: {len(result.entities)} entities, {len(result.relations)} relations")
                if result.subgraph:
                    print(
                        f"   Graph structure: {result.subgraph.number_of_nodes()} nodes, {result.subgraph.number_of_edges()} edges")

        return results

    def search_relations(self,
                         relation_type: str,
                         source_entity: Optional[str] = None,
                         target_entity: Optional[str] = None) -> List[KGSearchResult]:
        """
        Search for relations in the knowledge graph

        Args:
            relation_type: Type of relation (e.g., "founded", "works_at")
            source_entity: Optional source entity filter
            target_entity: Optional target entity filter

        Returns:
            List of KG search results
        """
        if not self.enable_kg:
            print("❌ Knowledge graph not enabled")
            return []

        print(f"🔍🔗 Searching relations for: '{relation_type}'")
        if source_entity:
            print(f"   📤 Source entity: {source_entity}")
        if target_entity:
            print(f"   📥 Target entity: {target_entity}")

        start_time = time.time()
        results = self.decoder.search_relations(relation_type, source_entity, target_entity)
        search_time = time.time() - start_time

        print(f"✅ Found {len(results)} relation results in {search_time:.3f}s")

        return results

    def get_entity_neighborhood(self,
                                entity_text: str,
                                depth: int = 1) -> Optional[KGSearchResult]:
        """
        Get the neighborhood of an entity up to a certain depth

        Args:
            entity_text: Entity to explore
            depth: How many hops from the entity (1-3 recommended)

        Returns:
            KG search result with expanded neighborhood
        """
        if not self.enable_kg:
            print("❌ Knowledge graph not enabled")
            return None

        print(f"🔍🌐 Exploring neighborhood of: '{entity_text}' (depth: {depth})")

        start_time = time.time()
        result = self.decoder.get_entity_neighborhood(entity_text, depth)
        search_time = time.time() - start_time

        if result:
            print(
                f"✅ Found neighborhood with {len(result.entities)} entities, {len(result.relations)} relations in {search_time:.3f}s")
            if result.subgraph:
                print(f"   Graph: {result.subgraph.number_of_nodes()} nodes, {result.subgraph.number_of_edges()} edges")
        else:
            print(f"❌ No neighborhood found for '{entity_text}'")

        return result

    def get_kg_statistics(self) -> Dict:
        """Get comprehensive knowledge graph statistics"""
        if not self.enable_kg:
            return {"enabled": False}

        print("📊 Analyzing knowledge graph statistics...")
        start_time = time.time()
        stats = self.decoder.get_kg_statistics()
        analysis_time = time.time() - start_time

        print(f"✅ Statistics computed in {analysis_time:.3f}s")
        return stats

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
                        show_vectors: bool = False,
                        show_kg_context: bool = True) -> None:
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

            # Show KG context if available
            if show_kg_context and 'kg_context' in result:
                kg_context = result['kg_context']
                if kg_context['entities'] or kg_context['relations']:
                    print(f"🔗 Knowledge Graph Context:")

                    if kg_context['entities']:
                        entities_text = ", ".join([f"{e['text']} ({e['label']})" for e in kg_context['entities'][:3]])
                        if len(kg_context['entities']) > 3:
                            entities_text += f" +{len(kg_context['entities']) - 3} more"
                        print(f"   👤 Entities: {entities_text}")

                    if kg_context['relations']:
                        relations_text = ", ".join([f"{r['relation_type']}" for r in kg_context['relations'][:3]])
                        if len(kg_context['relations']) > 3:
                            relations_text += f" +{len(kg_context['relations']) - 3} more"
                        print(f"   🔗 Relations: {relations_text}")

            print(f"📖 Content:")
            print(f"   {text}")

            if i < len(results):
                print("-" * 60)

    def display_kg_results(self, results: List[KGSearchResult], show_graph: bool = False) -> None:
        """Display knowledge graph search results"""
        if not results:
            print("❌ No KG results found!")
            return

        print(f"\n🔗 Knowledge Graph Results ({len(results)} found):")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n🔗 KG Result #{i} (Fragment: {result.fragment_id})")

            # Show entities
            if result.entities:
                print(f"👤 Entities ({len(result.entities)}):")
                for entity in result.entities[:5]:  # Show first 5
                    print(f"   • {entity['text']} ({entity['label']}) - Confidence: {entity['confidence']:.2f}")
                if len(result.entities) > 5:
                    print(f"   ... and {len(result.entities) - 5} more entities")

            # Show relations
            if result.relations:
                print(f"🔗 Relations ({len(result.relations)}):")
                for relation in result.relations[:5]:  # Show first 5
                    print(f"   • {relation['relation_type']} - Confidence: {relation['confidence']:.2f}")
                    print(f"     Context: {relation['context'][:100]}...")
                if len(result.relations) > 5:
                    print(f"   ... and {len(result.relations) - 5} more relations")

            # Show graph structure if requested
            if show_graph and result.subgraph:
                print(f"🌐 Graph Structure:")
                print(f"   Nodes: {result.subgraph.number_of_nodes()}")
                print(f"   Edges: {result.subgraph.number_of_edges()}")

                # Show some connections
                if result.subgraph.number_of_edges() > 0:
                    print(f"   Sample connections:")
                    for edge in list(result.subgraph.edges(data=True))[:3]:
                        source, target, data = edge
                        source_node = result.subgraph.nodes[source]
                        target_node = result.subgraph.nodes[target]
                        print(f"     {source_node.get('text', source)} → {target_node.get('text', target)}")

            if i < len(results):
                print("-" * 60)

    def display_kg_stats(self, stats: Dict) -> None:
        """Display knowledge graph statistics"""
        if not stats.get('enabled', False):
            print("🔗 Knowledge Graph: Not enabled")
            return

        print(f"\n📊 Knowledge Graph Statistics:")
        print("=" * 60)
        print(f"📚 Total entities: {stats['total_entities']:,}")
        print(f"🔗 Total relations: {stats['total_relations']:,}")
        print(f"🏷️  Entity types: {stats['unique_entity_types']}")
        print(f"↔️  Relation types: {stats['unique_relation_types']}")
        print(f"📦 KG fragments: {stats['kg_fragments']}")
        print(f"📊 Coverage: {stats['coverage_percentage']:.1f}% of chunks have KG data")

        # Show top entity types
        if 'entity_type_distribution' in stats:
            print(f"\n👤 Top Entity Types:")
            entity_types = sorted(stats['entity_type_distribution'].items(), key=lambda x: x[1], reverse=True)
            for entity_type, count in entity_types[:10]:
                print(f"   {entity_type}: {count:,}")

        # Show top relation types
        if 'relation_type_distribution' in stats:
            print(f"\n🔗 Top Relation Types:")
            relation_types = sorted(stats['relation_type_distribution'].items(), key=lambda x: x[1], reverse=True)
            for relation_type, count in relation_types[:10]:
                print(f"   {relation_type}: {count:,}")

        print("=" * 60)

    def interactive_search(self):
        """Start interactive search session with KG support"""
        print("\n🤖 Interactive Wikipedia Search with Knowledge Graph")
        print("Ask any question about topics in Wikipedia!")
        print("Now with entity relationships and deep inference!")
        print("\nCommands:")
        print("  /topics - List available topics")
        print("  /topic <name> - Search by specific topic")
        print("  /entity <text> - Search for entities")
        print("  /relation <type> - Search for relations")
        print("  /neighborhood <entity> - Explore entity neighborhood")
        print("  /kg-stats - Show knowledge graph statistics")
        print("  /hybrid <query> --entity <entity> - Hybrid search")
        print("  /cache - Show cache statistics")
        print("  /info - Show database information")
        print("  /help - Show this help")
        print("  quit - Exit")
        print("-" * 70)

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

                # Regular search - check if it looks like a hybrid query
                if '--entity' in query or '--relation' in query:
                    # Parse hybrid query
                    parts = query.split('--')
                    main_query = parts[0].strip()
                    entity_filter = None
                    relation_filter = None

                    for part in parts[1:]:
                        if part.startswith('entity '):
                            entity_filter = part[7:].strip()
                        elif part.startswith('relation '):
                            relation_filter = part[9:].strip()

                    results = self.hybrid_search(main_query, entity_filter, relation_filter, top_k=5)
                else:
                    # Regular vector search
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

                        # Show KG context if available
                        if 'kg_context' in result and (
                                result['kg_context']['entities'] or result['kg_context']['relations']):
                            kg = result['kg_context']
                            if kg['entities']:
                                entities_str = ", ".join([e['text'] for e in kg['entities'][:2]])
                                print(f"   👤 Entities: {entities_str}")
                            if kg['relations']:
                                relations_str = ", ".join([r['relation_type'] for r in kg['relations'][:2]])
                                print(f"   🔗 Relations: {relations_str}")

                    if len(results) > 3:
                        print(f"\n   ... and {len(results) - 3} more results")

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

        elif cmd == '/entity' and len(parts) > 1:
            entity_text = ' '.join(parts[1:])
            results = self.search_entities(entity_text)
            if results:
                self.display_kg_results(results[:2])  # Show first 2 results

        elif cmd == '/relation' and len(parts) > 1:
            relation_type = parts[1]
            results = self.search_relations(relation_type)
            if results:
                self.display_kg_results(results[:2])  # Show first 2 results

        elif cmd == '/neighborhood' and len(parts) > 1:
            entity_text = ' '.join(parts[1:])
            result = self.get_entity_neighborhood(entity_text, depth=2)
            if result:
                self.display_kg_results([result], show_graph=True)

        elif cmd == '/kg-stats':
            if self.enable_kg:
                stats = self.get_kg_statistics()
                self.display_kg_stats(stats)
            else:
                print("❌ Knowledge graph not enabled")

        elif cmd == '/hybrid' and len(parts) > 1:
            # Parse hybrid command: /hybrid query --entity entity_name --relation relation_type
            query_parts = []
            entity_filter = None
            relation_filter = None

            i = 1
            while i < len(parts):
                if parts[i] == '--entity' and i + 1 < len(parts):
                    entity_filter = parts[i + 1]
                    i += 2
                elif parts[i] == '--relation' and i + 1 < len(parts):
                    relation_filter = parts[i + 1]
                    i += 2
                else:
                    query_parts.append(parts[i])
                    i += 1

            if query_parts:
                query = ' '.join(query_parts)
                results = self.hybrid_search(query, entity_filter, relation_filter, top_k=3)
                if results:
                    self.display_results(results[:3], show_kg_context=True)

        elif cmd == '/cache':
            print(f"\n💾 Cache Statistics:")
            print(f"   Vector cache: {len(self.decoder.memory_cache)} fragments")
            if hasattr(self.decoder, 'kg_memory_cache'):
                print(f"   KG cache: {len(self.decoder.kg_memory_cache)} fragments")
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
            print("  /topic <n> - Search by specific topic")
            print("  /entity <text> - Search for entities in knowledge graph")
            print("  /relation <type> - Search for relations (e.g., 'founded', 'works_at')")
            print("  /neighborhood <entity> - Explore entity neighborhood")
            print("  /kg-stats - Show knowledge graph statistics")
            print("  /hybrid <query> --entity <entity> --relation <relation> - Hybrid search")
            print("  /cache - Show cache statistics")
            print("  /info - Show database information")
            print("  /help - Show this help")
            print("  quit - Exit the search")
            print("\nHybrid search examples:")
            print("  /hybrid machine learning --entity OpenAI")
            print("  /hybrid AI research --relation founded")
            print("  /hybrid neural networks --entity Google --relation develops")

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
    """Command line interface for Wikipedia search with KG support"""
    parser = argparse.ArgumentParser(
        description="Search Wikipedia vector database with natural language queries and knowledge graph support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic vector search
  python wikipedia_search.py "machine learning algorithms"

  # Hybrid search (vector + knowledge graph)
  python wikipedia_search.py "artificial intelligence" --entity "OpenAI" --relation "founded"

  # Entity search
  python wikipedia_search.py --search-entity "Elon Musk" --entity-type "PERSON"

  # Relation search
  python wikipedia_search.py --search-relation "founded" --source-entity "Google"

  # Entity neighborhood exploration
  python wikipedia_search.py --entity-neighborhood "Tesla" --depth 2

  # Interactive mode (recommended!)
  python wikipedia_search.py --interactive

  # Knowledge graph statistics
  python wikipedia_search.py --kg-stats

  # Custom URLs with KG disabled
  python wikipedia_search.py "neural networks" \\
    --mp4 "https://your-bucket.r2.dev/wikipedia.mp4" \\
    --no-kg

  # Model server usage (ultra-fast!)
  python wikipedia_search.py "machine learning" --use-server

Default R2 URLs:
  MP4: {DEFAULT_MP4_URL}
  Manifest: {DEFAULT_MANIFEST_URL}
  Faiss: {DEFAULT_FAISS_URL}
        """
    )

    # Query argument
    parser.add_argument("query", nargs='?',
                        help="Search query (optional in interactive mode)")

    # File location arguments
    parser.add_argument("--mp4", default=DEFAULT_MP4_URL,
                        help=f"URL to MP4 vector database")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_URL,
                        help=f"URL to manifest file")
    parser.add_argument("--faiss", default=DEFAULT_FAISS_URL,
                        help=f"URL to Faiss index file")

    # Search mode arguments
    parser.add_argument("--search-entity", metavar="TEXT",
                        help="Search for entities containing this text")
    parser.add_argument("--entity-type", metavar="TYPE",
                        help="Filter entities by type (PERSON, ORG, GPE, etc.)")
    parser.add_argument("--search-relation", metavar="TYPE",
                        help="Search for relations of this type")
    parser.add_argument("--source-entity", metavar="TEXT",
                        help="Filter relations by source entity")
    parser.add_argument("--target-entity", metavar="TEXT",
                        help="Filter relations by target entity")
    parser.add_argument("--entity-neighborhood", metavar="TEXT",
                        help="Explore entity neighborhood")
    parser.add_argument("--depth", type=int, default=2,
                        help="Neighborhood search depth (default: 2)")

    # Hybrid search arguments
    parser.add_argument("--entity", metavar="TEXT",
                        help="Filter hybrid search by entity")
    parser.add_argument("--relation", metavar="TYPE",
                        help="Filter hybrid search by relation type")

    # Model and search parameters
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Embedding model name")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to return")
    parser.add_argument("--topic",
                        help="Filter results by topic")
    parser.add_argument("--min-similarity", type=float, default=0.0,
                        help="Minimum similarity threshold")

    # Knowledge graph options
    parser.add_argument("--no-kg", action="store_true",
                        help="Disable knowledge graph functionality")
    parser.add_argument("--kg-stats", action="store_true",
                        help="Show knowledge graph statistics and exit")

    # Performance and caching
    parser.add_argument("--cache-dir",
                        help="Directory for persistent cache")
    parser.add_argument("--cache-size", type=int, default=100,
                        help="Memory cache size in fragments")
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
    parser.add_argument("--show-graph", action="store_true",
                        help="Show graph structure in KG results")
    parser.add_argument("--max-text", type=int, default=300,
                        help="Maximum text length to display")

    # Model server options
    parser.add_argument("--use-server", action="store_true",
                        help="Use persistent model server for instant encoding")
    parser.add_argument("--server-host", default="localhost",
                        help="Model server host")
    parser.add_argument("--server-port", type=int, default=8888,
                        help="Model server port")

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
        print("🌐 Wikipedia Vector Search with Knowledge Graph Support")

        with WikipediaSearchEngine(
                mp4_path=args.mp4,
                manifest_path=args.manifest,
                faiss_path=args.faiss,
                model_name=args.model,
                cache_dir=args.cache_dir,
                cache_size=args.cache_size,
                enable_prefetching=not args.no_prefetch,
                max_retries=args.max_retries,
                timeout=args.timeout,
                use_model_server=args.use_server,
                enable_kg=not args.no_kg
        ) as search_engine:

            # Handle different modes
            if args.kg_stats:
                if search_engine.enable_kg:
                    stats = search_engine.get_kg_statistics()
                    search_engine.display_kg_stats(stats)
                else:
                    print("❌ Knowledge graph not enabled")
                return 0

            if args.list_topics:
                topics = search_engine.get_available_topics()
                print(f"\n🏷️  Available topics ({len(topics)}):")
                for i, topic in enumerate(topics, 1):
                    print(f"  {i:3d}. {topic}")
                return 0

            if args.search_entity:
                results = search_engine.search_entities(args.search_entity, args.entity_type)
                search_engine.display_kg_results(results, show_graph=args.show_graph)
                return 0

            if args.search_relation:
                results = search_engine.search_relations(
                    args.search_relation,
                    args.source_entity,
                    args.target_entity
                )
                search_engine.display_kg_results(results, show_graph=args.show_graph)
                return 0

            if args.entity_neighborhood:
                result = search_engine.get_entity_neighborhood(args.entity_neighborhood, args.depth)
                if result:
                    search_engine.display_kg_results([result], show_graph=True)
                return 0

            if args.interactive:
                search_engine.interactive_search()
                return 0

            if not args.query:
                print("❌ Please provide a search query or use a specific search mode")
                print("Example: python wikipedia_search.py \"machine learning\"")
                print("Or use: python wikipedia_search.py --interactive")
                return 1

            # Perform search based on type
            if args.entity or args.relation:
                # Hybrid search
                print(f"\n🎯 Performing hybrid search...")
                results = search_engine.hybrid_search(
                    query=args.query,
                    entity_filter=args.entity,
                    relation_filter=args.relation,
                    top_k=args.top_k,
                    show_performance=args.show_performance
                )
            else:
                # Regular vector search
                print(f"\n🎯 Performing vector search...")
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
                show_vectors=args.show_vectors,
                show_kg_context=True
            )

            return 0

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required dependencies:")
        print("  pip install sentence-transformers faiss-cpu numpy requests networkx")
        print("And that the enhanced decoder with KG support is available!")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.show_performance:
            import traceback
            traceback.print_exc()
        return 1


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


def check_model_server(host="localhost", port=8888):
    """Check if model server is running"""
    try:
        from ragged.video.model_server import ModelClient
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


# Quick demo functions
def demo_hybrid_search():
    """Quick demo of hybrid search functionality"""
    print("🧪 Demo: Hybrid Wikipedia Search with Knowledge Graph")

    try:
        with WikipediaSearchEngine(enable_kg=True) as engine:
            queries = [
                ("machine learning", "OpenAI", None),
                ("artificial intelligence", None, "founded"),
                ("neural networks", "Google", "develops")
            ]

            for query, entity_filter, relation_filter in queries:
                print(f"\n--- Hybrid Search: {query} ---")
                if entity_filter:
                    print(f"    Entity filter: {entity_filter}")
                if relation_filter:
                    print(f"    Relation filter: {relation_filter}")

                results = engine.hybrid_search(query, entity_filter, relation_filter, top_k=2)

                if results:
                    for i, result in enumerate(results, 1):
                        metadata = result['metadata']
                        text = metadata.get('text', '')[:100] + "..."
                        print(f"{i}. {metadata.get('source', 'Unknown')} (sim: {result['similarity']:.3f})")
                        print(f"   {text}")

                        # Show KG context
                        if 'kg_context' in result:
                            kg = result['kg_context']
                            if kg['entities']:
                                print(f"   👤 Entities: {[e['text'] for e in kg['entities'][:2]]}")
                            if kg['relations']:
                                print(f"   🔗 Relations: {[r['relation_type'] for r in kg['relations'][:2]]}")
                else:
                    print("   No results found")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_kg_exploration():
    """Quick demo of knowledge graph exploration"""
    print("🧪 Demo: Knowledge Graph Exploration")

    try:
        with WikipediaSearchEngine(enable_kg=True) as engine:
            # Show KG stats
            print("\n--- Knowledge Graph Statistics ---")
            stats = engine.get_kg_statistics()
            if stats.get('enabled'):
                print(f"Entities: {stats['total_entities']:,}")
                print(f"Relations: {stats['total_relations']:,}")
                print(f"Entity types: {list(stats.get('entity_types', {}).keys())[:5]}")
                print(f"Relation types: {list(stats.get('relation_types', {}).keys())[:5]}")

            # Entity search example
            print("\n--- Entity Search: 'Tesla' ---")
            entity_results = engine.search_entities("Tesla")
            for result in entity_results[:1]:
                print(f"Found {len(result.entities)} entities, {len(result.relations)} relations")

            # Relation search example
            print("\n--- Relation Search: 'founded' ---")
            relation_results = engine.search_relations("founded")
            for result in relation_results[:1]:
                print(f"Found {len(result.relations)} founded relations")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


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

# Uncomment to run quick demos
# demo_basic_search()
# demo_hybrid_search()
# demo_kg_exploration()