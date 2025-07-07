"""
RAG connector with FAISS and HuggingFace embeddings
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .models import RAGChunk


class RAGConnector:
    """Simple RAG connector using FAISS + HuggingFace embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "rag_index"):
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.chunks: List[Dict[str, Any]] = []
        
        # Load existing index if available
        self._load_index()
    
    def index_documents(self, documents_path: str, chunk_size: int = 512, overlap: int = 50):
        """Index documents from a directory"""
        documents_path = Path(documents_path)
        
        if not documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        # Find all .txt and .md files
        files = list(documents_path.glob("*.txt")) + list(documents_path.glob("*.md"))
        
        if not files:
            raise ValueError(f"No .txt or .md files found in {documents_path}")
        
        print(f"Indexing {len(files)} files...")
        
        for file_path in files:
            print(f"Processing: {file_path.name}")
            self._index_file(file_path, chunk_size, overlap)
        
        self._save_index()
        print(f"Indexed {len(self.chunks)} chunks total")
    
    def _index_file(self, file_path: Path, chunk_size: int, overlap: int):
        """Index a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Warning: Could not read {file_path} as UTF-8, skipping")
            return
        
        # Split into chunks
        chunks = self._split_text(content, chunk_size, overlap)
        
        # Generate embeddings
        embeddings = self.encoder.encode(chunks)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunk metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                'content': chunk,
                'source': str(file_path),
                'chunk_id': len(self.chunks)
            })
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundaries
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:  # Only if we don't lose too much
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def query(self, query: str, top_k: int = 3) -> List[RAGChunk]:
        """Query the RAG index"""
        if self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk_data = self.chunks[idx]
                results.append(RAGChunk(
                    content=chunk_data['content'],
                    source=chunk_data['source'],
                    similarity=float(similarity)
                ))
        
        return results
    
    def _save_index(self):
        """Save FAISS index and chunk metadata"""
        if self.index.ntotal > 0:
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        
        with open(self.index_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def _load_index(self):
        """Load existing FAISS index and chunk metadata"""
        index_file = self.index_path / "faiss.index"
        chunks_file = self.index_path / "chunks.pkl"
        
        if index_file.exists() and chunks_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"Loaded existing index with {len(self.chunks)} chunks")
            except Exception as e:
                print(f"Warning: Could not load existing index ({e}), starting fresh")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.chunks = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG index statistics"""
        return {
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'sources': list(set(chunk['source'] for chunk in self.chunks))
        }
