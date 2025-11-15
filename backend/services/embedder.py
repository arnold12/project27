"""
Embedding generation service using OpenAI.
"""
from typing import List, Optional
import openai
from openai import OpenAI

from backend.core.config import settings


class EmbedderService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        """Initialize the embedder with OpenAI client."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector, or None if error
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        embeddings = []
        
        # OpenAI API supports batch processing
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            # Map responses to input order
            embedding_dict = {item.index: item.embedding for item in response.data}
            for idx in range(len(texts)):
                embeddings.append(embedding_dict.get(idx))
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            # Fallback to individual generation
            for text in texts:
                embeddings.append(self.generate_embedding(text))
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.
        
        Returns:
            Embedding dimension
        """
        # text-embedding-3-large has 3072 dimensions
        if "3-large" in self.model:
            return 3072
        # text-embedding-3-small has 1536 dimensions
        elif "3-small" in self.model:
            return 1536
        # Default for older models
        else:
            return 1536


# Global embedder instance
embedder_service = EmbedderService()

