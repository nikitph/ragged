from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class TextChunk:
    """Represents a processed text chunk with metadata"""
    text: str
    source: str
    chunk_id: int
    topic: Optional[str] = None
    word_count: int = 0
    token_count: int = 0
    timestamp: str = datetime.now().isoformat()