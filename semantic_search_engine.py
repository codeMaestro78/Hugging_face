"""
Semantic Search Engine with FAISS and Hugging Face Embeddings

This module provides a complete semantic search engine that uses:
- Hugging Face transformers for generating embeddings
- FAISS for efficient similarity search
- Support for multiple embedding models
- Batch processing and persistence
"""

import numpy as np
import faiss
import pickle
import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with score and metadata"""
    text: str
    score: float
    index: int
    metadata: Optional[Dict] = None


class HuggingFaceEmbedder:
    """
    Wrapper for Hugging Face embedding models with pooling strategies
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        pooling_strategy: str = "mean"
    ):
        """
        Initialize the embedder
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda', 'cpu', or None for auto)
            pooling_strategy: How to pool token embeddings ('mean', 'cls', 'max')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pooling_strategy = pooling_strategy
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            sample_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            sample_input = {k: v.to(self.device) for k, v in sample_input.items()}
            sample_output = self.model(**sample_input)
            self.embedding_dim = sample_output.last_hidden_state.shape[-1]
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to token embeddings"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output, attention_mask):
        """Use CLS token embedding"""
        return model_output[0][:, 0]  # CLS token is at position 0
    
    def _max_pooling(self, model_output, attention_mask):
        """Apply max pooling to token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        with torch.no_grad():
            for batch in tqdm(batches, desc="Encoding texts", disable=not show_progress):
                # Tokenize batch
                encoded_input = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                )
                
                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Get model output
                model_output = self.model(**encoded_input)
                
                # Apply pooling
                if self.pooling_strategy == "mean":
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                elif self.pooling_strategy == "cls":
                    embeddings = self._cls_pooling(model_output, encoded_input['attention_mask'])
                elif self.pooling_strategy == "max":
                    embeddings = self._max_pooling(model_output, encoded_input['attention_mask'])
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


class SemanticSearchEngine:
    """
    Complete semantic search engine using FAISS and Hugging Face embeddings
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat",
        device: Optional[str] = None,
        pooling_strategy: str = "mean"
    ):
        """
        Initialize the search engine
        
        Args:
            model_name: HuggingFace model for embeddings
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            device: Device for model inference
            pooling_strategy: Pooling strategy for embeddings
        """
        self.embedder = HuggingFaceEmbedder(
            model_name=model_name,
            device=device,
            pooling_strategy=pooling_strategy
        )
        
        self.index_type = index_type
        self.index = None
        self.texts = []
        self.metadata = []
        
        logger.info(f"Semantic search engine initialized with {model_name}")
    
    def _create_index(self, dimension: int, num_vectors: int) -> faiss.Index:
        """Create FAISS index based on type and data size"""
        
        if self.index_type == "flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            
        elif self.index_type == "ivf":
            # Inverted file index for faster approximate search
            nlist = min(int(np.sqrt(num_vectors)), 1000)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World for very fast search
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created {self.index_type} index with dimension {dimension}")
        return index
    
    def add_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ):
        """
        Add documents to the search index
        
        Args:
            texts: List of text documents
            metadata: Optional metadata for each document
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Adding {len(texts)} documents to index")
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = self._create_index(embeddings.shape[1], len(texts))
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings.astype('float32'))
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Store texts and metadata
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))
        
        logger.info(f"Added {len(texts)} documents. Total: {len(self.texts)}")
    
    def search(
        self,
        query: str,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or len(self.texts) == 0:
            logger.warning("No documents in index")
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Set search parameters for HNSW
        if self.index_type == "hnsw":
            self.index.hnsw.efSearch = max(k * 2, 50)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                if threshold is None or score >= threshold:
                    results.append(SearchResult(
                        text=self.texts[idx],
                        score=float(score),
                        index=int(idx),
                        metadata=self.metadata[idx]
                    ))
        
        return results
    
    def save(self, save_path: str):
        """Save the search engine to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save texts and metadata
        with open(save_path / "texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)
        
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save configuration
        config = {
            "model_name": self.embedder.model_name,
            "index_type": self.index_type,
            "pooling_strategy": self.embedder.pooling_strategy,
            "embedding_dim": self.embedder.embedding_dim
        }
        
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Search engine saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str, device: Optional[str] = None):
        """Load a search engine from disk"""
        load_path = Path(load_path)
        
        # Load configuration
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        engine = cls(
            model_name=config["model_name"],
            index_type=config["index_type"],
            device=device,
            pooling_strategy=config["pooling_strategy"]
        )
        
        # Load FAISS index
        engine.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load texts and metadata
        with open(load_path / "texts.json", "r", encoding="utf-8") as f:
            engine.texts = json.load(f)
        
        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            engine.metadata = json.load(f)
        
        logger.info(f"Search engine loaded from {load_path}")
        return engine


def main():
    """Example usage of the semantic search engine"""
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning teaches agents through rewards and penalties.",
        "Supervised learning uses labeled data for training.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning leverages pre-trained models for new tasks.",
        "Transformer models have revolutionized NLP.",
        "BERT is a bidirectional encoder representation from transformers."
    ]
    
    # Create search engine
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    # Add documents
    engine.add_documents(documents)
    
    # Search examples
    queries = [
        "neural networks and deep learning",
        "understanding human language",
        "learning without labels"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        results = engine.search(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.score:.4f}] {result.text}")
    
    # Save the engine
    engine.save("./search_engine_data")
    print("\nSearch engine saved successfully!")


if __name__ == "__main__":
    main()
