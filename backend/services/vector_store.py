"""
Vector store service for storing and retrieving embeddings.
Supports FAISS (local) and MongoDB Atlas Vector Search.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

from backend.core.config import settings
from backend.core.utils import ensure_directory, save_json, load_json


class VectorStore:
    """Service for storing and querying embeddings."""
    
    def __init__(self, embedding_dim: int = None):
        """Initialize the vector store."""
        self.db_type = settings.vector_db.lower()
        # Lazy import to avoid circular dependencies
        if embedding_dim is None:
            from backend.services.embedder import embedder_service
            self.embedding_dim = embedder_service.get_embedding_dimension()
        else:
            self.embedding_dim = embedding_dim
        
        if self.db_type == "faiss":
            self._init_faiss()
        elif self.db_type == "mongo":
            self._init_mongodb()
        else:
            raise ValueError(f"Unsupported vector DB type: {self.db_type}")
    
    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS is not available. Please install it: pip install faiss-cpu"
            )
        
        ensure_directory(os.path.dirname(settings.faiss_index_path))
        self.index_path = settings.faiss_index_path
        self.metadata_path = f"{self.index_path}_metadata.json"
        
        # Load or create index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.metadata = load_json(self.metadata_path)
        else:
            # Create new index (L2 distance)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = {}
    
    def _init_mongodb(self) -> None:
        """Initialize MongoDB connection."""
        if not PYMONGO_AVAILABLE:
            raise RuntimeError(
                "pymongo is not available. Please install it: pip install pymongo"
            )
        
        if not settings.mongodb_uri:
            raise ValueError("MongoDB URI is required for MongoDB vector store")
        
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db_name]
        self.collection: Collection = self.db[settings.mongodb_collection]
        
        # Create vector search index if it doesn't exist
        self._ensure_vector_index()
    
    def _ensure_vector_index(self) -> None:
        """Ensure MongoDB vector search index exists."""
        try:
            # Check if index exists
            indexes = self.collection.list_indexes()
            index_names = [idx["name"] for idx in indexes]
            
            if "vector_index" not in index_names:
                # Create vector search index
                self.db.command({
                    "createIndexes": settings.mongodb_collection,
                    "indexes": [{
                        "name": "vector_index",
                        "key": {"embedding": "vector"},
                        "vectorOptions": {
                            "type": "knnVector",
                            "dimensions": self.embedding_dim,
                            "similarity": "cosine"
                        }
                    }]
                })
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")
    
    def add_embeddings(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Add embeddings to the vector store.
        
        Args:
            document_id: Document ID
            chunks: List of chunk metadata dictionaries
            embeddings: List of embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        
        if self.db_type == "faiss":
            return self._add_to_faiss(document_id, chunks, embeddings)
        else:
            return self._add_to_mongodb(document_id, chunks, embeddings)
    
    def _add_to_faiss(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """Add embeddings to FAISS index."""
        try:
            # Convert to numpy array
            embedding_array = np.array(embeddings, dtype=np.float32)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(embedding_array)
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                idx = start_idx + i
                self.metadata[str(idx)] = {
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index", i),
                    "text": chunk["text"],
                    "text_length": chunk.get("text_length", len(chunk["text"])),
                    "token_count": chunk.get("token_count", 0),
                }
            
            # Save index and metadata
            faiss.write_index(self.index, self.index_path)
            save_json(self.metadata, self.metadata_path)
            
            return True
        except Exception as e:
            print(f"Error adding to FAISS: {e}")
            return False
    
    def _add_to_mongodb(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """Add embeddings to MongoDB."""
        try:
            documents = []
            for chunk, embedding in zip(chunks, embeddings):
                doc = {
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk["text"],
                    "text_length": chunk.get("text_length", len(chunk["text"])),
                    "token_count": chunk.get("token_count", 0),
                    "embedding": embedding,
                }
                documents.append(doc)
            
            self.collection.insert_many(documents)
            return True
        except Exception as e:
            print(f"Error adding to MongoDB: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            document_id: Optional filter by document ID
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        if self.db_type == "faiss":
            return self._search_faiss(query_embedding, top_k, document_id)
        else:
            return self._search_mongodb(query_embedding, top_k, document_id)
    
    def _search_faiss(
        self,
        query_embedding: List[float],
        top_k: int,
        document_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_array, top_k * 2)  # Get more to filter
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                metadata = self.metadata.get(str(idx))
                if not metadata:
                    continue
                
                # Filter by document_id if specified
                if document_id and metadata["document_id"] != document_id:
                    continue
                
                # Convert L2 distance to similarity (inverse)
                similarity = 1 / (1 + dist)
                
                results.append({
                    **metadata,
                    "similarity": float(similarity),
                    "distance": float(dist),
                })
                
                if len(results) >= top_k:
                    break
            
            return results
        except Exception as e:
            print(f"Error searching FAISS: {e}")
            return []
    
    def _search_mongodb(
        self,
        query_embedding: List[float],
        top_k: int,
        document_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search MongoDB vector index."""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": top_k * 2,
                        "limit": top_k
                    }
                }
            ]
            
            if document_id:
                pipeline.append({
                    "$match": {"document_id": document_id}
                })
            
            pipeline.append({
                "$project": {
                    "document_id": 1,
                    "chunk_id": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "text_length": 1,
                    "token_count": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            })
            
            results = list(self.collection.aggregate(pipeline))
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "document_id": result["document_id"],
                    "chunk_id": result["chunk_id"],
                    "chunk_index": result.get("chunk_index", 0),
                    "text": result["text"],
                    "text_length": result.get("text_length", 0),
                    "token_count": result.get("token_count", 0),
                    "similarity": result.get("score", 0.0),
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching MongoDB: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk metadata
        """
        if self.db_type == "faiss":
            return self._get_chunks_from_faiss(document_id)
        else:
            return self._get_chunks_from_mongodb(document_id)
    
    def _get_chunks_from_faiss(self, document_id: str) -> List[Dict[str, Any]]:
        """Get chunks from FAISS metadata."""
        chunks = []
        for idx, metadata in self.metadata.items():
            if metadata["document_id"] == document_id:
                chunks.append(metadata)
        return sorted(chunks, key=lambda x: x.get("chunk_index", 0))
    
    def _get_chunks_from_mongodb(self, document_id: str) -> List[Dict[str, Any]]:
        """Get chunks from MongoDB."""
        results = self.collection.find(
            {"document_id": document_id},
            {"embedding": 0}  # Exclude embeddings to save space
        ).sort("chunk_index", 1)
        
        return list(results)


# Global vector store instance
vector_store = VectorStore()

