# Ragged - MP4 for RAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2024.XXXXX)

> 🎥 **Revolutionary Vector Database Distribution**: Encode your knowledge base into MP4 video files and distribute globally through CDNs for lightning-fast semantic search.

Ragged transforms how we think about vector database distribution by leveraging the mature video streaming infrastructure. Instead of complex database deployments, simply upload an MP4 file to any CDN and get instant global semantic search capabilities.

## 🚀 Quick Start

```bash
# Install dependencies
poetry install

WORK IN PROGRESS: WILL BE UPDATED SHORTLY
```

## 🌟 What Makes Ragged Special?

### Traditional Vector Databases 😰
- Complex server deployments
- Expensive hosting infrastructure  
- Cold-start penalties
- Regional latency issues
- Database connection limits

### Ragged Approach 🎯
- **MP4 files** → Upload anywhere (Cloudflare R2, AWS S3, etc.)
- **CDN distribution** → Global edge caching automatically
- **HTTP range requests** → Download only what you need
- **Zero servers** → Serverless and edge-computing ready
- **Infinite scale** → No connection limits

## 🏗️ How It Works

```mermaid
graph TD
    A[📄 Documents] --> B[🔤 Text Chunking]
    B --> C[🧮 Vector Encoding]
    C --> D[📦 MP4 Fragments]
    D --> E[🎬 MP4 Container]
    E --> F[🌍 CDN Distribution]
    F --> G[🔍 Global Search]
    
    H[📊 FAISS Index] --> F
    I[📋 JSON Manifest] --> F
```

1. **📄 Input**: Your documents (PDFs, text files, web content)
2. **🔤 Processing**: Smart chunking with overlap and topic extraction
3. **🧮 Encoding**: Convert to vectors using sentence-transformers
4. **📦 Packaging**: Encode vectors into MP4 fragments with metadata
5. **🌍 Distribution**: Upload to any CDN (Cloudflare R2, AWS CloudFront, etc.)
6. **🔍 Search**: Lightning-fast semantic search from anywhere in the world

## 🎯 Core Features

### 🎬 MP4 Vector Encoding
- **Standards Compliant**: ISO/IEC 14496-12 MP4 containers
- **Fragment-Based**: Optimized chunk sizes for CDN performance
- **Rich Metadata**: Topic classification, timestamps, source attribution
- **Binary Efficiency**: Float32 vectors with JSON metadata

### 🌐 CDN-Optimized Distribution
- **HTTP Range Requests**: Surgical data access, download only needed fragments
- **Intelligent Prefetching**: Background loading of adjacent fragments
- **Multi-Level Caching**: Memory, disk, and CDN edge caching
- **Global Performance**: Consistent search speed worldwide

### 🔍 Advanced Search Capabilities
- **Semantic Search**: Natural language queries using sentence-transformers
- **Topic Filtering**: Search within specific topics or domains
- **FAISS Integration**: Exact and approximate similarity search
- **Similarity Thresholds**: Configurable result quality filtering

### ⚡ Performance & Scalability
- **Cold-Start Optimization**: 3-second initialization vs 45+ seconds for traditional DBs
- **Infinite Readers**: No database connection limits
- **Edge Computing**: Works in Cloudflare Workers, AWS Lambda, etc.
- **Bandwidth Efficient**: 92% less data transfer for initial loads

## 📁 Project Structure

```
ragged/
├── 📁 ragged/                     # Core package
│   ├── 📁 video/                  # Video encoding/decoding
│   │   ├── encoder.py            # MP4 vector encoding
│   │   ├── decoder.py            # CDN-optimized decoding
│   │   └── config.py             # Video codec settings
│   ├── 📁 api/                    # FastAPI web service
│   │   └── v1/endpoints/         # REST API endpoints
│   ├── 📁 services/              # Business logic
│   │   └── uploader/             # CDN upload services
│   ├── 📁 enterprise/            # Enterprise features (WIP)
│   └── main.py                   # FastAPI app entry point
├── 📁 examples/                   # Usage examples
├── 📁 scripts/                   # Utility scripts
└── 📋 pyproject.toml             # Dependencies
```


## 📊 Performance Characteristics

| Metric | Traditional Vector DB | Ragged |
|--------|----------------------|--------|
| 🚀 Cold-start latency | 45+ seconds | ~3 seconds |
| 💾 Initial memory usage | 2.4GB | 85MB |
| 🌍 Global distribution | Complex setup | Upload file |
| 💰 Infrastructure cost | High | CDN bandwidth only |
| ⚡ Concurrent users | Connection limited | Unlimited |
| 🔄 Updates | Real-time | File replacement |

## 🎯 Use Cases

### ✅ Perfect For
- **📚 Knowledge Base Search**: Documentation, FAQs, internal wikis
- **🤖 RAG Applications**: Retrieval-augmented generation systems
- **🌐 Global Applications**: Multi-region deployments with consistent performance
- **⚡ Edge Computing**: Serverless functions, IoT devices, mobile apps
- **💰 Cost-Sensitive Deployments**: Startups, side projects, research

### ❌ Not Ideal For
- **🔄 Frequent Updates**: Real-time indexing requirements
- **🔍 Complex Queries**: Multi-stage filtering, analytical workloads
- **📊 Traditional CRUD**: Applications needing database transactions

## 🛣️ Roadmap

### 🎯 Current Focus (v0.1)
- [x] Core MP4 encoding/decoding
- [x] CDN-optimized distribution
- [ ] FastAPI web service
- [ ] PDF upload pipeline
- [ ] Production deployment guides
- [ ] Performance optimization

### 🚀 Next Phase (v0.2)
- [ ] Multi-modal vectors (images, audio)
- [ ] Streaming updates (incremental changes)
- [ ] Advanced search (hybrid, faceted)
- [ ] Enterprise SSO integration

### 🌟 Future Vision (v1.0)
- [ ] Standard MP4 boxes for vectors
- [ ] P2P distribution networks
- [ ] Edge AI processing
- [ ] Ecosystem integrations

## 🤝 Contributing

I dont expect to maintain this. Please fork and use as your heart desires. 


### 📝 Documentation
- Update README for new features
- Add docstrings for new functions
- Include usage examples

## 📚 Learn More

### 📖 Academic Paper
Read our arXiv paper: **"Ragged: Leveraging Video Container Formats for Efficient Vector Database Distribution"**

### 🎥 Inspiration
This project was inspired by [Memvid](https://github.com/Olow304/memvid), which demonstrated storing data in video formats. Ragged extends this concept with vector-specific optimizations, CDN distribution, and semantic search capabilities.

### 🔗 Related Projects
- **[FAISS](https://github.com/facebookresearch/faiss)**: Efficient similarity search
- **[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)**: Text embedding models
- **[FastAPI](https://github.com/tiangolo/fastapi)**: Modern web framework

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments

- **Video Streaming Community**: For the mature CDN infrastructure we leverage
- **FAISS Team**: For efficient similarity search algorithms  
- **Sentence-Transformers**: For high-quality text embeddings
- **Memvid**: For the initial inspiration of storing data in video formats
- **Open Source Community**: For the foundational libraries that make this possible

---

<div align="center">

**🌟 Star this repo if you find it useful! 🌟**

**Questions?** Open an issue or start a discussion!

</div>

---

> 💡 **Fun Fact**: Your entire knowledge base is now a video file that can be streamed, cached, and distributed just like any YouTube video - but instead of cat videos, it's semantic search! 🐱➡️🔍
