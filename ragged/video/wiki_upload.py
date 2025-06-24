import os
import json
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import time

from ragged.video import VectorMP4Encoder
from ragged.video.encoder import TextVectorPipeline


# Import your existing classes (assuming they're in the same file or imported)
# from your_module import TextVectorPipeline, VectorMP4Encoder, create_text_vector_mp4


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
    Complete pipeline: Wikipedia → Vectors → MP4

    This is like converting Wikipedia into a searchable mathematical format
    that can be distributed as efficiently as streaming video content.
    """

    print("🚀 Starting Wikipedia to MP4 Vector Encoding Pipeline")
    print("=" * 60)

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

    # Step 2: Initialize text processing pipeline
    print("\n🔧 Step 2: Initializing text processing pipeline...")

    pipeline = TextVectorPipeline(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_dim=384  # MiniLM model dimension
    )

    # Step 3: Process documents into vectors
    print("\n🧮 Step 3: Converting text to vectors...")
    vector_start = time.time()

    vectors, metadata = pipeline.process_documents(documents)

    vector_time = time.time() - vector_start
    print(f"⏱️  Vector encoding completed in {vector_time:.1f} seconds")
    print(f"📊 Generated {len(vectors)} vectors from {len(documents)} articles")

    if len(vectors) == 0:
        print("❌ No vectors generated!")
        return

    # Step 4: Encode vectors to MP4
    print("\n📹 Step 4: Encoding vectors to MP4...")
    mp4_start = time.time()

    encoder = VectorMP4Encoder(
        vector_dim=384,
        chunk_size=vector_batch_size
    )

    encoder.add_vectors(vectors, metadata)
    encoder.encode_and_upload(output_path)

    mp4_time = time.time() - mp4_start
    total_time = time.time() - start_time

    print(f"⏱️  MP4 encoding completed in {mp4_time:.1f} seconds")
    print(f"🎉 Total pipeline time: {total_time:.1f} seconds")

    # Step 5: Generate summary report
    print("\n📋 Final Summary:")
    print("=" * 40)
    print(f"📚 Source: Wikipedia {language.upper()} ({date})")
    print(f"📄 Articles processed: {len(documents)}")
    print(f"🧩 Text chunks created: {len(vectors)}")
    print(f"🎯 Vector dimensions: {vectors.shape[1] if len(vectors) > 0 else 0}")
    print(f"💾 Output file: {output_path}")
    print(f"📦 File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")

    # Additional files created
    manifest_path = output_path.replace('.mp4', '_manifest.json')
    faiss_path = output_path.replace('.mp4', '_faiss.index')

    print(f"📋 Manifest: {manifest_path}")
    if os.path.exists(faiss_path):
        print(f"🔍 Search index: {faiss_path}")

    print("\n✅ Wikipedia MP4 vector database created successfully!")
    return output_path


def main():
    """Command line interface for the Wikipedia MP4 encoder"""
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia articles to MP4 vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 1000 English Wikipedia articles
  python wikipedia_mp4.py --language en --max-articles 1000

  # Process 5000 Spanish articles with custom output
  python wikipedia_mp4.py --language es --max-articles 5000 --output wikipedia_es.mp4

  # Process with different chunking strategy
  python wikipedia_mp4.py --chunk-size 256 --chunk-overlap 25
        """
    )

    parser.add_argument("--language", default="en",
                        help="Wikipedia language code (default: en)")
    parser.add_argument("--date", default="20231101",
                        help="Wikipedia dump date YYYYMMDD (default: 20231101)")
    parser.add_argument("--output", default="wikipedia_vectors.mp4",
                        help="Output MP4 file path (default: wikipedia_vectors.mp4)")
    parser.add_argument("--max-articles", type=int, default=1000,
                        help="Maximum articles to process (default: 1000)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Text chunk size in tokens (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Overlap between chunks in tokens (default: 50)")
    parser.add_argument("--vector-batch-size", type=int, default=1000,
                        help="Vectors per MP4 fragment (default: 1000)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model (default: all-MiniLM-L6-v2)")

    args = parser.parse_args()

    try:
        create_wikipedia_mp4(
            language=args.language,
            date=args.date,
            output_path=args.output,
            max_articles=args.max_articles,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vector_batch_size=args.vector_batch_size,
            model_name=args.model
        )
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()


# Quick start function for Jupyter notebooks
def quick_start_demo():
    """
    Quick demo function for testing - processes just 100 articles
    Perfect for testing the pipeline before running on larger datasets
    """
    print("🧪 Running quick demo with 100 Wikipedia articles...")

    try:
        output_path = create_wikipedia_mp4(
            language="en",
            max_articles=100,
            output_path="demo_wikipedia.mp4",
            chunk_size=256,  # Smaller chunks for demo
        )
        print(f"\n🎉 Demo completed! Check out: {output_path}")
        return output_path
    except Exception as e:
        print(f"Demo failed: {e}")
        return None