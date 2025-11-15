"""
Tests for embedding generation.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from backend.services.embedder import EmbedderService
from backend.core.config import settings


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch("backend.services.embedder.OpenAI") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        
        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536  # Default embedding dimension
        client_instance.embeddings.create.return_value = mock_response
        
        yield client_instance


def test_embedder_initialization():
    """Test embedder service initialization."""
    with patch("backend.services.embedder.OpenAI"):
        service = EmbedderService()
        assert service.model == settings.openai_embedding_model


def test_generate_embedding(mock_openai_client):
    """Test single embedding generation."""
    service = EmbedderService()
    service.client = mock_openai_client
    
    text = "Sample text for embedding"
    embedding = service.generate_embedding(text)
    
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == 1536
    mock_openai_client.embeddings.create.assert_called_once()


def test_generate_embeddings_batch(mock_openai_client):
    """Test batch embedding generation."""
    service = EmbedderService()
    service.client = mock_openai_client
    
    # Mock batch response
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(index=0, embedding=[0.1] * 1536),
        MagicMock(index=1, embedding=[0.2] * 1536),
    ]
    mock_openai_client.embeddings.create.return_value = mock_response
    
    texts = ["Text 1", "Text 2"]
    embeddings = service.generate_embeddings_batch(texts)
    
    assert len(embeddings) == 2
    assert all(emb is not None for emb in embeddings)


def test_get_embedding_dimension():
    """Test embedding dimension detection."""
    service = EmbedderService()
    
    # Test for different models
    service.model = "text-embedding-3-large"
    assert service.get_embedding_dimension() == 3072
    
    service.model = "text-embedding-3-small"
    assert service.get_embedding_dimension() == 1536
    
    service.model = "text-embedding-ada-002"
    assert service.get_embedding_dimension() == 1536

