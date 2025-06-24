import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from ragged.services.vector_encoding import VectorMP4Encoder
from ragged.services.text_processing import TextVectorPipeline


class WikipediaVectorizer:
    """Complete pipeline for converting Wikipedia to vector MP4 format"""

    def __init__(self):
        self.documents = []
        self.vectors = None
        self.metadata = []

    def download_wikipedia_sample(
            self,
            language: str = "en",
            date: str = "20231101",
            max_articles: Optional[int] = 1000,
            min_text_length: int = 500,
            streaming: bool = True
    ) -> List[Dict[str, str]]:
        """Download and process Wikipedia articles"""
        print(f"Loading Wikipedia {language} ({date})")

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
            raise

        documents = []
        stats = {
            'processed': 0,
            'skipped_short': 0,
            'skipped_redirect': 0
        }

        with tqdm(desc="Processing articles") as pbar:
            for article in wiki_data:
                if max_articles and stats['processed'] >= max_articles:
                    break

                title = article.get("title", "Unknown")
                text = article.get("text", "")
                url = article.get("url", "")

                if len(text) < min_text_length:
                    stats['skipped_short'] += 1
                    continue

                if text.strip().lower().startswith("#redirect"):
                    stats['skipped_redirect'] += 1
                    continue

                documents.append({
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
                })
                stats['processed'] += 1
                pbar.update(1)
                pbar.set_postfix(stats)

        print(f"\nProcessed {stats['processed']} articles")
        self.documents = documents
        return documents

    def process_to_vectors(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            chunk_size: int = 512,
            chunk_overlap: int = 50
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Convert documents to vectors"""
        if not self.documents:
            raise ValueError("No documents to process")

        pipeline = TextVectorPipeline(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_dim=384
        )

        self.vectors, self.metadata = pipeline.process_documents(self.documents)
        return self.vectors, self.metadata

    def encode_to_mp4(
            self,
            output_path: str,
            vector_batch_size: int = 1000
    ) -> Dict[str, str]:
        """Encode vectors to MP4 format with all artifacts"""
        if self.vectors is None:
            raise ValueError("Vectors not generated yet")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize encoder with our optimized settings
        encoder = VectorMP4Encoder(
            vector_dim=self.vectors.shape[1],
            chunk_size=vector_batch_size
        )

        # Add vectors with metadata
        encoder.add_vectors(self.vectors, self.metadata)

        # Encode to MP4 (this handles the main file)
        mp4_path = output_path
        encoder.encode_to_mp4(mp4_path)

        # Get generated paths
        base_path = os.path.splitext(mp4_path)[0]
        manifest_path = f"{base_path}_manifest.json"
        faiss_path = f"{base_path}_faiss.index"

        # Verify all files were created
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(f"MP4 file not created at {mp4_path}")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not created at {manifest_path}")

        # Return all artifact paths
        return {
            "mp4": mp4_path,
            "manifest": manifest_path,
            "faiss_index": faiss_path if os.path.exists(faiss_path) else None
        }


def create_wikipedia_mp4(
        language: str = "en",
        date: str = "20231101",
        output_path: str = "wikipedia_vectors.mp4",
        max_articles: int = 1000,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_batch_size: int = 1000,
        model_name: str = "all-MiniLM-L6-v2"
) -> dict[str, str]:
    """Complete pipeline with timing and reporting"""
    print("🚀 Starting Wikipedia to MP4 Vector Encoding Pipeline")
    start_time = time.time()
    vectorizer = WikipediaVectorizer()

    # Step 1: Download
    print("📥 Step 1: Downloading Wikipedia...")
    vectorizer.download_wikipedia_sample(
        language=language,
        date=date,
        max_articles=max_articles
    )

    # Step 2: Vectorize
    print("\n🔧 Step 2: Processing text to vectors...")
    vectorizer.process_to_vectors(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Step 3: Encode
    print("\n📹 Step 3: Encoding to MP4...")
    output = vectorizer.encode_to_mp4(
        output_path=output_path,
        vector_batch_size=vector_batch_size
    )

    # Reporting
    total_time = time.time() - start_time
    print(f"\n🎉 Completed in {total_time:.1f} seconds")
    print(f"📦 Output: {output}")
    return output


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia to MP4 vector database",
        epilog="Example: python wikipedia_mp4.py --language en --max-articles 1000"
    )

    parser.add_argument("--language", default="en", help="Wikipedia language code")
    parser.add_argument("--date", default="20231101", help="Wikipedia dump date")
    parser.add_argument("--output", default="wikipedia_vectors.mp4", help="Output file")
    parser.add_argument("--max-articles", type=int, default=1000, help="Max articles")
    parser.add_argument("--chunk-size", type=int, default=512, help="Text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--vector-batch-size", type=int, default=1000, help="Fragment size")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")

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
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()