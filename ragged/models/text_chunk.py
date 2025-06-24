import logging
from dataclasses import dataclass
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TextChunk:
    """Represents a processed text chunk with metadata"""
    text: str
    source: str
    chunk_id: int # A unique ID within the source document
    word_count: int
    token_count: int
    timestamp: str
    topic: Optional[str] = None