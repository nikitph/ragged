from .encoder import VideoEncoder
from .retriever import VideoRetriever

# Create video memory from text chunks
chunks = ["Important fact 1", "Important fact 2", "Historical event details"]
encoder = VideoEncoder()
encoder.add_chunks(chunks)
encoder.build_video("memory.mp4", "memory_index.json")

retriever = VideoRetriever("knowledge_base.mp4", "knowledge_index.json")

# Semantic search
results = retriever.search("machine learning algorithms", top_k=5)
for chunk, score in results:
    print(f"Score: {score:.3f} | {chunk[:100]}...")

# Get context window
context = retriever.get_context("explain neural networks", max_tokens=2000)
print(context)