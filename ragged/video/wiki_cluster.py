import os
import json
import argparse
import time
from typing import List, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm

# --- Import the new VectorPipelineManager ---
# This assumes the refactored classes are in a file named `refactored_pipeline.py`
# You would need to have that file in the same directory or in your Python path.
from ragged.services.vector_pipeline_manager import VectorPipelineManager


# --- Helper function for data acquisition (remains the same) ---

def download_wikipedia_sample(
        language: str = "en",
        date: str = "20231101",
        max_articles: Optional[int] = 1000,
        min_text_length: int = 500,
        streaming: bool = True
) -> List[Dict[str, str]]:
    """
    Download and process Wikipedia articles from Hugging Face.
    This function is now solely focused on data retrieval.
    """
    print(f"Loading Wikipedia dataset: {language}, {date}")
    print(f"Target articles: {max_articles if max_articles else 'all'}")

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"{date}.{language}",
            streaming=streaming,
            trust_remote_code=True
        )
        wiki_data = dataset["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Available configurations might include dates like: 20231101, 20231201")
        raise

    documents = []
    processed_count = 0
    skipped_short = 0
    skipped_redirect = 0

    pbar = tqdm(total=max_articles, desc="Downloading Wikipedia articles")

    for article in wiki_data:
        if max_articles and processed_count >= max_articles:
            break

        text = article.get("text", "")
        if len(text) < min_text_length:
            skipped_short += 1
            continue
        if text.strip().lower().startswith("#redirect"):
            skipped_redirect += 1
            continue

        # We only need 'text' and 'source' for the pipeline manager.
        # The manager handles all further metadata creation.
        doc = {
            "text": text,
            "source": f"wikipedia_{language}_{article.get('title', 'Unknown')}",
        }
        documents.append(doc)
        processed_count += 1
        pbar.update(1)

    pbar.close()

    print(f"\nDataset Summary:")
    print(f"✓ Processed articles: {processed_count}")
    print(f"✗ Skipped (too short/redirect): {skipped_short + skipped_redirect}")
    if documents:
        avg_len = sum(len(d['text']) for d in documents) // len(documents)
        print(f"📊 Average article length: {avg_len} chars")

    return documents


# --- Main pipeline function, now refactored ---

def create_wikipedia_mp4(
        language: str,
        date: str,
        output_path: str,
        max_articles: int,
        model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        n_clusters_per_doc: int
):
    """
    Complete pipeline: Wikipedia -> Vectors -> Clustered MP4.
    This function orchestrates the download and then hands off to the VectorPipelineManager.
    """
    print("🚀 Starting Wikipedia to MP4 Vector Encoding Pipeline")
    print("=" * 60)

    # Step 1: Download Wikipedia data
    print("📥 Step 1: Downloading Wikipedia articles...")
    start_time = time.time()
    documents = download_wikipedia_sample(
        language=language,
        date=date,
        max_articles=max_articles
    )
    download_time = time.time() - start_time
    print(f"⏱️  Download completed in {download_time:.1f} seconds")

    if not documents:
        print("❌ No documents to process! Exiting.")
        return

    # Step 2: Initialize and run the Vector Pipeline Manager
    print("\n⚙️  Step 2: Processing data with VectorPipelineManager...")
    pipeline_start_time = time.time()

    # The manager now handles all the complex steps:
    # chunking, embedding, clustering, indexing, and writing files.
    manager = VectorPipelineManager(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        n_clusters_per_doc=n_clusters_per_doc
    )

    # Execute the entire pipeline with one call
    manager.run_hierarchical(documents, output_path)

    pipeline_time = time.time() - pipeline_start_time
    total_time = time.time() - start_time

    print(f"⏱️  Pipeline processing completed in {pipeline_time:.1f} seconds")
    print(f"🎉 Total run time: {total_time:.1f} seconds")

    # Step 3: Generate summary report from the created files
    print("\n📋 Final Summary:")
    print("=" * 40)
    print(f"📚 Source: Wikipedia {language.upper()} ({date})")
    print(f"📄 Articles processed: {len(documents)}")

    manifest_path = output_path.replace('.mp4', '_manifest.json')
    faiss_path = output_path.replace('.mp4', '_faiss.index')

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        meta = manifest.get("metadata", {})
        print(f"🧩 Text chunks created: {meta.get('total_vectors', 'N/A')}")
        print(f"📦 Clusters (fragments): {meta.get('total_clusters', 'N/A')}")
        print(f"🎯 Vector dimensions: {meta.get('vector_dim', 'N/A')}")

    if os.path.exists(output_path):
        print(f"💾 Output file: {output_path}")
        print(f"📦 File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")

    if os.path.exists(manifest_path):
        print(f"📋 Manifest: {manifest_path}")
    if os.path.exists(faiss_path):
        print(f"🔍 Search index: {faiss_path}")

    print("\n✅ Wikipedia MP4 vector database created successfully!")
    return output_path


def main():
    """Command line interface for the Wikipedia MP4 encoder"""
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia articles to a clustered MP4 vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 1000 English Wikipedia articles into 50 clusters
  python wiki.py --language en --max-articles 1000 --n-clusters 50

  # Process 5000 Spanish articles with custom output
  python wiki.py --language es --max-articles 5000 --n-clusters 200 --output wikipedia_es.mp4
        """
    )
    parser.add_argument("--language", default="en", help="Wikipedia language code (default: en)")
    parser.add_argument("--date", default="20231101", help="Wikipedia dump date YYYYMMDD (default: 20231101)")
    parser.add_argument("--output", default="wikipedia_vectors.mp4", help="Output MP4 file path")
    parser.add_argument("--max-articles", type=int, default=1000, help="Maximum articles to process")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")

    # --- Arguments for the new architecture ---
    parser.add_argument("--chunk-size", type=int, default=512, help="Text chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks in tokens")
    parser.add_argument(
        "--clusters-per-doc",
        type=int,
        default=5,
        dest="n_clusters_per_doc",  # Store in n_clusters_per_doc
        help="Number of sub-topic clusters to find within each article (for IVFPQ index)."
    )

    args = parser.parse_args()

    try:
        create_wikipedia_mp4(
            language=args.language,
            date=args.date,
            output_path=args.output,
            max_articles=args.max_articles,
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            n_clusters_per_doc=args.n_clusters_per_doc  # Pass the new argument
        )
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()