"""
Pytest configuration for Ragged tests
"""

import pytest
import os
import tempfile
from pathlib import Path

# Set environment variable to suppress tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@pytest.fixture(scope="session")
def test_data_dir():
    """Shared test data directory"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)