import os
import json
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import time

# Import the enhanced encoder and pipeline with KG support
from ragged.video.encoder import TextVectorPipeline, VectorMP4Encoder, create_text_vector_mp4


def download_wikipedia_sample(
        language: str = "en",
        date: str = "20231101",
        max_articles: Optional[int] = 1000,
        min_text_length: int = 500,
        streaming: bool = True
) -> List[Dict[str, str]]:
    """
    Download and process Wikipedia articles from Hugging Face

    Args:
        language: Wikipedia language code (e.g., 'en', 'es', 'fr')
        date: Wikipedia dump date (YYYYMMDD format)
        max_articles: Maximum number of articles to process (None for all)
        min_text_length: Minimum article length in characters
        streaming: Use streaming to avoid downloading entire dataset

    Returns:
        List of documents with 'text' and 'source' keys
    """
    print(f"Loading Wikipedia dataset: {language}, {date}")
    print(f"Target articles: {max_articles if max_articles else 'all'}")

    # Load the dataset - this is like opening a massive digital encyclopedia
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"{date}.{language}",
            streaming=streaming,
            trust_remote_code=True
        )

        # Get the train split (Wikipedia articles)
        wiki_data = dataset["train"]

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Available configurations might include dates like: 20231101, 20231201")
        print("Try: dataset = load_dataset('wikimedia/wikipedia', '20231101.en')")
        raise

    documents = []
    processed_count = 0
    skipped_short = 0
    skipped_redirect = 0

    print("Processing articles...")

    # Process articles with progress tracking
    pbar = tqdm(desc="Processing Wikipedia articles")

    for article in wiki_data:
        if max_articles and processed_count >= max_articles:
            break

        # Extract article data
        title = article.get("title", "Unknown")
        text = article.get("text", "")
        url = article.get("url", "")

        # Skip short articles or redirects
        if len(text) < min_text_length:
            skipped_short += 1
            continue

        if text.strip().lower().startswith("#redirect"):
            skipped_redirect += 1
            continue

        # Create document entry
        # Think of each article as a book chapter in our digital library
        doc = {
            "text": text,
            "source": f"wikipedia_{language}_{title}",
            "metadata": {
                "title": title,
                "url": url,
                "language": language,
                "date": date,
                "char_count": len(text),
                "word_count": len(text.split())
            }
        }

        documents.append(doc)
        processed_count += 1
        pbar.update(1)
        pbar.set_description(f"Processed: {processed_count}, Skipped: {skipped_short + skipped_redirect}")

    pbar.close()

    print(f"\nDataset Summary:")
    print(f"✓ Processed articles: {processed_count}")
    print(f"✗ Skipped (too short): {skipped_short}")
    print(f"✗ Skipped (redirects): {skipped_redirect}")
    print(
        f"📊 Average article length: {sum(len(d['text']) for d in documents) // len(documents) if documents else 0} chars")

    return documents


def create_wikipedia_mp4_with_kg(
        language: str = "en",
        date: str = "20231101",
        output_path: str = "wikipedia_vectors.mp4",
        max_articles: int = 1000,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_batch_size: int = 1000,
        model_name: str = "all-MiniLM-L6-v2",
        enable_kg: bool = True,
        kg_max_fragment_size: int = 100
):
    """
    Complete pipeline: Wikipedia → Vectors + Knowledge Graph → MP4

    This is like converting Wikipedia into a searchable mathematical format
    WITH entity relationships that can be distributed as efficiently as
    streaming video content.

    Args:
        enable_kg: Whether to enable knowledge graph extraction
        kg_max_fragment_size: Maximum entities per KG fragment
    """

    print("🚀 Starting Wikipedia to MP4 Vector + Knowledge Graph Encoding Pipeline")
    if enable_kg:
        print("🔗 Knowledge Graph extraction: ENABLED")
    else:
        print("🔗 Knowledge Graph extraction: DISABLED")
    print("=" * 70)

    # Step 1: Download Wikipedia data
    print("📥 Step 1: Downloading Wikipedia articles...")
    start_time = time.time()

    documents = download_wikipedia_sample(
        language=language,
        date=date,
        max_articles=max_articles,
        min_text_length=500,
        streaming=True
    )

    download_time = time.time() - start_time
    print(f"⏱️  Download completed in {download_time:.1f} seconds")

    if not documents:
        print("❌ No documents to process!")
        return

    # Step 2: Initialize enhanced text processing pipeline
    print("\n🔧 Step 2: Initializing enhanced text processing pipeline...")

    pipeline = TextVectorPipeline(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_dim=384  # MiniLM model dimension
    )

    # Step 3: Process documents into vectors and text chunks
    print("\n🧮 Step 3: Converting text to vectors...")
    vector_start = time.time()

    # Use the enhanced process_documents that returns text chunks for KG processing
    vectors, metadata, text_chunks = pipeline.process_documents(documents)

    vector_time = time.time() - vector_start
    print(f"⏱️  Vector encoding completed in {vector_time:.1f} seconds")
    print(f"📊 Generated {len(vectors)} vectors from {len(documents)} articles")
    print(f"📝 Created {len(text_chunks)} text chunks for processing")

    if len(vectors) == 0:
        print("❌ No vectors generated!")
        return

    # Step 4: Initialize enhanced MP4 encoder with KG support
    print(f"\n📹 Step 4: Initializing enhanced MP4 encoder...")
    if enable_kg:
        print("   🔗 Knowledge graph extraction will be performed")

    mp4_start = time.time()

    encoder = VectorMP4Encoder(
        vector_dim=384,
        chunk_size=vector_batch_size,
        enable_kg=enable_kg
    )

    # Step 5: Add vectors to encoder
    print("\n🧩 Step 5: Adding vectors to encoder...")
    encoder.add_vectors(vectors, metadata)

    # Step 6: Process knowledge graph data (if enabled)
    if enable_kg:
        print("\n🔗 Step 6: Processing knowledge graph data...")
        kg_start = time.time()

        encoder.add_knowledge_graph_data(text_chunks)

        kg_time = time.time() - kg_start
        print(f"⏱️  Knowledge graph processing completed in {kg_time:.1f} seconds")

        # Show KG statistics
        kg_stats = encoder.manifest["metadata"]["knowledge_graph"]
        print(f"📊 KG Statistics:")
        print(f"   👤 Entities extracted: {kg_stats.get('total_entities', 0):,}")
        print(f"   🔗 Relations extracted: {kg_stats.get('total_relations', 0):,}")
        print(f"   📦 KG fragments: {len(kg_stats.get('fragments', []))}")

        if kg_stats.get('entity_types'):
            top_entity_types = sorted(kg_stats['entity_types'].items(),
                                      key=lambda x: x[1], reverse=True)[:5]
            print(f"   🏷️  Top entity types: {', '.join([f'{t}({c})' for t, c in top_entity_types])}")
    else:
        print("\n⏭️  Step 6: Skipping knowledge graph processing (disabled)")

    # Step 7: Encode to MP4
    print(f"\n🎬 Step 7: Encoding to MP4...")
    encoder.encode_to_mp4(output_path)

    mp4_time = time.time() - mp4_start
    total_time = time.time() - start_time

    print(f"⏱️  MP4 encoding completed in {mp4_time:.1f} seconds")
    print(f"🎉 Total pipeline time: {total_time:.1f} seconds")

    # Step 8: Generate comprehensive summary report
    print("\n📋 Final Summary:")
    print("=" * 50)
    print(f"📚 Source: Wikipedia {language.upper()} ({date})")
    print(f"📄 Articles processed: {len(documents)}")
    print(f"🧩 Text chunks created: {len(text_chunks)}")
    print(f"🔢 Vectors generated: {len(vectors)}")
    print(f"🎯 Vector dimensions: {vectors.shape[1] if len(vectors) > 0 else 0}")

    if enable_kg:
        kg_meta = encoder.manifest["metadata"]["knowledge_graph"]
        print(f"🔗 Knowledge Graph:")
        print(f"   👤 Total entities: {kg_meta.get('total_entities', 0):,}")
        print(f"   🔗 Total relations: {kg_meta.get('total_relations', 0):,}")
        print(f"   📦 KG fragments: {len(kg_meta.get('fragments', []))}")

        # Calculate KG coverage
        total_vectors = encoder.manifest["metadata"]["total_vectors"]
        kg_coverage = len(kg_meta.get('fragments', [])) * 100 / len(encoder.fragments) if encoder.fragments else 0
        print(f"   📊 KG coverage: {kg_coverage:.1f}% of fragments")

    print(f"💾 Output file: {output_path}")

    if os.path.exists(output_path):
        print(f"📦 File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")

    # Additional files created
    manifest_path = output_path.replace('.mp4', '_manifest.json')
    faiss_path = output_path.replace('.mp4', '_faiss.index')

    print(f"📋 Manifest: {manifest_path}")
    if os.path.exists(faiss_path):
        faiss_size = os.path.getsize(faiss_path) / (1024 * 1024)
        print(f"🔍 Search index: {faiss_path} ({faiss_size:.1f} MB)")

    print(
        f"\n✅ Wikipedia MP4 vector database with {'Knowledge Graph' if enable_kg else 'vectors only'} created successfully!")

    # Show usage suggestions
    print(f"\n🚀 Usage suggestions:")
    print(f"   # Basic search:")
    print(f"   python wikipedia_search.py \"machine learning\"")
    if enable_kg:
        print(f"   # Hybrid search with KG:")
        print(f"   python wikipedia_search.py \"AI research\" --entity \"OpenAI\"")
        print(f"   # Interactive mode with KG:")
        print(f"   python wikipedia_search.py --interactive")

    return output_path


def create_wikipedia_mp4(
        language: str = "en",
        date: str = "20231101",
        output_path: str = "wikipedia_vectors.mp4",
        max_articles: int = 1000,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_batch_size: int = 1000,
        model_name: str = "all-MiniLM-L6-v2"
):
    """
    Backward compatibility wrapper - creates MP4 without KG by default
    """
    return create_wikipedia_mp4_with_kg(
        language=language,
        date=date,
        output_path=output_path,
        max_articles=max_articles,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_batch_size=vector_batch_size,
        model_name=model_name,
        enable_kg=False  # Default to false for backward compatibility
    )


def main():
    """Command line interface for the Wikipedia MP4 encoder with KG support"""
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia articles to MP4 vector database with optional Knowledge Graph support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 1000 English Wikipedia articles with Knowledge Graph
  python wikipedia_mp4.py --language en --max-articles 1000 --enable-kg

  # Process 5000 Spanish articles with custom output (vectors only)
  python wikipedia_mp4.py --language es --max-articles 5000 --output wikipedia_es.mp4

  # Process with different chunking strategy and KG enabled
  python wikipedia_mp4.py --chunk-size 256 --chunk-overlap 25 --enable-kg

  # Large dataset with KG optimizations
  python wikipedia_mp4.py --max-articles 10000 --enable-kg --kg-fragment-size 150

  # Quick test with small dataset
  python wikipedia_mp4.py --max-articles 100 --enable-kg --output test.mp4
        """
    )

    # Dataset parameters
    parser.add_argument("--language", default="en",
                        help="Wikipedia language code (default: en)")
    parser.add_argument("--date", default="20231101",
                        help="Wikipedia dump date YYYYMMDD (default: 20231101)")
    parser.add_argument("--max-articles", type=int, default=1000,
                        help="Maximum articles to process (default: 1000)")

    # Output parameters
    parser.add_argument("--output", default="wikipedia_vectors.mp4",
                        help="Output MP4 file path (default: wikipedia_vectors.mp4)")

    # Text processing parameters
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Text chunk size in tokens (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Overlap between chunks in tokens (default: 50)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model (default: all-MiniLM-L6-v2)")

    # Vector encoding parameters
    parser.add_argument("--vector-batch-size", type=int, default=1000,
                        help="Vectors per MP4 fragment (default: 1000)")

    # Knowledge Graph parameters
    parser.add_argument("--enable-kg", action="store_true",
                        help="Enable knowledge graph extraction (default: disabled)")
    parser.add_argument("--kg-fragment-size", type=int, default=100,
                        help="Maximum entities per KG fragment (default: 100)")
    parser.add_argument("--no-kg", action="store_true",
                        help="Explicitly disable knowledge graph (for clarity)")

    # Performance parameters
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress information")

    args = parser.parse_args()

    # Handle KG enable/disable logic
    enable_kg = args.enable_kg and not args.no_kg

    if args.verbose:
        print(f"Configuration:")
        print(f"  Language: {args.language}")
        print(f"  Date: {args.date}")
        print(f"  Max articles: {args.max_articles}")
        print(f"  Knowledge Graph: {'Enabled' if enable_kg else 'Disabled'}")
        print(f"  Model: {args.model}")
        print(f"  Chunk size: {args.chunk_size}")
        print(f"  Output: {args.output}")

    try:
        create_wikipedia_mp4_with_kg(
            language=args.language,
            date=args.date,
            output_path=args.output,
            max_articles=args.max_articles,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vector_batch_size=args.vector_batch_size,
            model_name=args.model,
            enable_kg=enable_kg,
            kg_max_fragment_size=args.kg_fragment_size
        )
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


def quick_start_demo():
    """
    Quick demo function for testing - processes just 100 articles with KG
    Perfect for testing the pipeline before running on larger datasets
    """
    print("🧪 Running quick demo with 100 Wikipedia articles + Knowledge Graph...")

    try:
        output_path = create_wikipedia_mp4_with_kg(
            language="en",
            max_articles=100,
            output_path="demo_wikipedia.mp4",
            chunk_size=256,  # Smaller chunks for demo
            enable_kg=True,  # Enable KG for demo
            kg_max_fragment_size=50  # Smaller KG fragments for demo
        )
        print(f"\n🎉 Demo completed! Check out: {output_path}")
        print(f"\nTry searching with:")
        print(f"  python wikipedia_search.py --interactive")
        return output_path
    except Exception as e:
        print(f"Demo failed: {e}")
        return None


def quick_start_demo_vectors_only():
    """
    Quick demo function for testing vectors only (no KG)
    For testing basic functionality
    """
    print("🧪 Running quick demo with 100 Wikipedia articles (vectors only)...")

    try:
        output_path = create_wikipedia_mp4_with_kg(
            language="en",
            max_articles=100,
            output_path="demo_wikipedia_vectors_only.mp4",
            chunk_size=256,
            enable_kg=False
        )
        print(f"\n🎉 Demo completed! Check out: {output_path}")
        return output_path
    except Exception as e:
        print(f"Demo failed: {e}")
        return None


# Convenience functions for different use cases
def create_small_kg_database(language="en", articles=1000, output="small_wikipedia_kg.mp4"):
    """Create a small knowledge graph database for testing"""
    return create_wikipedia_mp4_with_kg(
        language=language,
        max_articles=articles,
        output_path=output,
        enable_kg=True,
        chunk_size=256,
        kg_max_fragment_size=75
    )


def create_large_kg_database(language="en", articles=10000, output="large_wikipedia_kg.mp4"):
    """Create a large knowledge graph database for production"""
    return create_wikipedia_mp4_with_kg(
        language=language,
        max_articles=articles,
        output_path=output,
        enable_kg=True,
        chunk_size=512,
        vector_batch_size=1500,
        kg_max_fragment_size=150
    )


def create_multilingual_demo():
    """Create small databases in multiple languages"""
    languages = ["en", "es", "fr", "de"]
    results = {}

    for lang in languages:
        print(f"\n🌍 Creating {lang.upper()} database...")
        try:
            output_path = create_wikipedia_mp4_with_kg(
                language=lang,
                max_articles=500,
                output_path=f"wikipedia_{lang}_kg.mp4",
                enable_kg=True
            )
            results[lang] = output_path
            print(f"✅ {lang.upper()} database created: {output_path}")
        except Exception as e:
            print(f"❌ Failed to create {lang.upper()} database: {e}")
            results[lang] = None

    return results


if __name__ == "__main__":
    main()

# Example notebook cells:
"""
# Quick test in Jupyter:
quick_start_demo()

# Create small KG database:
create_small_kg_database("en", 2000, "my_wikipedia_kg.mp4")

# Create vectors-only database:
quick_start_demo_vectors_only()

# Create multilingual databases:
create_multilingual_demo()
"""