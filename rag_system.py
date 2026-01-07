"""
RAG System for Knowledge Base Retrieval
Handles document embedding, indexing, and semantic search using FAISS
"""

import os
import json
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval-Augmented Generation system for semantic search over knowledge base"""
    
    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG system
        
        Args:
            knowledge_base_path: Path to knowledge base documents
            model_name: HuggingFace model for embeddings
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.knowledge_base_path = knowledge_base_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # FAISS index
        self.faiss_index = None
        self.documents = []
        self.chunks = []
        self.metadata = []
        
        # Cache paths
        self.index_cache = "data/faiss_index.bin"
        self.metadata_cache = "data/metadata.json"
        self.chunks_cache = "data/chunks.pkl"
    
    def load_documents(self) -> Dict[str, str]:
        """Load all documents from knowledge base directory"""
        documents = {}
        kb_path = Path(self.knowledge_base_path)
        
        if not kb_path.exists():
            logger.warning(f"Knowledge base path not found: {self.knowledge_base_path}")
            return documents
        
        for file_path in kb_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents[file_path.name] = f.read()
                logger.info(f"Loaded document: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def chunk_text(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            source: Source document name
        
        Returns:
            List of (chunk, metadata) tuples
        """
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if len(chunk.strip()) > 0:
                metadata = {
                    'source': source,
                    'start_char': i,
                    'end_char': min(i + self.chunk_size, len(text)),
                    'chunk_size': len(chunk)
                }
                chunks.append((chunk, metadata))
        
        return chunks
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build FAISS index from knowledge base
        
        Args:
            force_rebuild: Force rebuild even if cache exists
        
        Returns:
            True if successful, False otherwise
        """
        # Check if we can load from cache
        if (not force_rebuild and 
            os.path.exists(self.index_cache) and 
            os.path.exists(self.metadata_cache)):
            logger.info("Loading FAISS index from cache")
            return self._load_from_cache()
        
        logger.info("Building FAISS index from scratch")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to index")
            return False
        
        # Create chunks
        all_chunks = []
        for doc_name, doc_text in documents.items():
            chunks = self.chunk_text(doc_text, doc_name)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return False
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Extract texts and metadata
        texts = [chunk[0] for chunk in all_chunks]
        self.metadata = [chunk[1] for chunk in all_chunks]
        self.chunks = texts
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(embeddings)
        
        # Save to cache
        logger.info("Saving index to cache")
        self._save_to_cache()
        
        logger.info("FAISS index built successfully")
        return True
    
    def _save_to_cache(self):
        """Save index and metadata to cache"""
        os.makedirs("data", exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, self.index_cache)
        
        # Save metadata
        with open(self.metadata_cache, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save chunks
        with open(self.chunks_cache, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info("Cache saved successfully")
    
    def _load_from_cache(self) -> bool:
        """Load index and metadata from cache"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(self.index_cache)
            
            # Load metadata
            with open(self.metadata_cache, 'r') as f:
                self.metadata = json.load(f)
            
            # Load chunks
            with open(self.chunks_cache, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Loaded {len(self.chunks)} chunks from cache")
            return True
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant chunks
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of relevant chunks with metadata and scores
        """
        if self.faiss_index is None:
            logger.warning("FAISS index not initialized")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convert distances to similarity scores
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No result
                continue
            
            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1.0 / (1.0 + dist)
            
            if similarity < similarity_threshold:
                continue
            
            result = {
                'rank': i + 1,
                'chunk': self.chunks[idx],
                'similarity_score': float(similarity),
                'distance': float(dist),
                'metadata': self.metadata[idx]
            }
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query}")
        return results
    
    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Get formatted context for LLM from search results
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
        
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant documents found."
        
        context = "Retrieved Knowledge Base Documents:\n\n"
        for result in results:
            source = result['metadata'].get('source', 'Unknown')
            score = result['similarity_score']
            chunk = result['chunk']
            
            context += f"[Source: {source} | Similarity: {score:.2%}]\n"
            context += f"{chunk}\n"
            context += "-" * 80 + "\n\n"
        
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            'total_chunks': len(self.chunks),
            'total_sources': len(set(m.get('source', '') for m in self.metadata)),
            'embedding_dimension': self.embedding_dim,
            'model': self.model.get_sentence_embedding_dimension(),
            'index_type': 'FAISS IndexFlatL2'
        }


def initialize_rag() -> RAGSystem:
    """Initialize and build RAG system"""
    rag = RAGSystem()
    rag.build_index()
    return rag


if __name__ == "__main__":
    # Test RAG system
    rag = initialize_rag()
    
    print("RAG System Stats:")
    print(json.dumps(rag.get_stats(), indent=2))
    
    # Test queries
    test_queries = [
        "What are the work hours?",
        "How much annual leave do employees get?",
        "What is the code of conduct?",
        "Travel policy information",
        "Internship compensation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = rag.search(query, top_k=3)
        for result in results:
            print(f"\nRank {result['rank']} | Similarity: {result['similarity_score']:.2%}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Content: {result['chunk'][:200]}...")
