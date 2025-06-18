# Ragged - Video-based RAG as a Service

## Quick Start

```bash
# Install dependencies
poetry install

# Run development server
poetry run python ragged/main.py
```

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `POST /api/v1/videos/upload` - Upload documents
- `POST /api/v1/search` - Search knowledge base

## TODO

- [ ] Implement video processing
- [ ] Add real search functionality
- [ ] Set up database migrations
- [ ] Add authentication
- [ ] Implement multi-tenancy
- [ ] Add enterprise features

## Architecture

This project is structured for enterprise multi-tenancy but starts with simple implementations.
