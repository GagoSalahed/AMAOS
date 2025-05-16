"""
Semantic Memory implementation for AMAOS.

This module provides vector-based semantic memory capabilities with FAISS,
including fallback to traditional keyword search when embedding fails.
"""

import json
import logging
import os
import pickle
import hashlib
import time
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Set, cast, TYPE_CHECKING, Protocol, TypeVar, Callable
from typing_extensions import TypedDict, Literal, NotRequired

# Global flag variables
FAISS_AVAILABLE = False
EMBEDDINGS_AVAILABLE = False

# Add path to stubs directory for type checking
import sys
from pathlib import Path

# Add stubs directory to path for type checking
if TYPE_CHECKING:
    stubs_path = Path(__file__).parent / "stubs"
    if stubs_path.exists() and str(stubs_path) not in sys.path:
        sys.path.insert(0, str(stubs_path))

# Handle conditional imports
try:
    import faiss  # type: ignore # missing-imports
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer  # type: ignore # missing-imports
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from amaos.memory.memory import Memory, MemoryType
from amaos.utils.errors import MemoryError, ErrorContext
from pydantic import BaseModel, Field

class IndexType(str, Enum):
    """Type of index to use for semantic search."""
    FLAT = "flat"        # Exact search, slower but most accurate
    IVF = "ivf"         # Inverted file index, faster with some accuracy tradeoff
    HNSW = "hnsw"       # Hierarchical Navigable Small World, very fast with good accuracy


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    
    model_name: str = "all-MiniLM-L6-v2"  # Default lightweight model
    dimension: int = 384  # Dimension of the embeddings (depends on model)
    normalize: bool = True  # Whether to normalize vectors
    batch_size: int = 32  # Batch size for embedding
    max_seq_length: int = 256  # Maximum sequence length


class IndexConfig(BaseModel):
    """Configuration for FAISS index."""
    
    index_type: str = "Flat"  # Type of FAISS index (Flat, IVF, HNSW)
    metric_type: str = "L2"  # Metric type (L2, IP, etc.)
    nlist: int = 100  # Number of cells for IVF indices
    m: int = 16  # Number of neighbors for HNSW indices
    ef_construction: int = 80  # Construction factor for HNSW
    ef_search: int = 20  # Search factor for HNSW


class SemanticMemoryConfig(BaseModel):
    """Configuration for semantic memory."""
    
    # Embedding model configuration
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # Index configuration
    index: IndexConfig = Field(default_factory=IndexConfig)
    # Storage
    cache_dir: str = "./semantic_cache"  # Directory to store indices
    save_interval: int = 60  # How often to save the index (in seconds)
    # Search behavior
    max_results: int = 10  # Maximum number of results to return
    fallback_to_keyword: bool = True  # Fallback to keyword search if embedding fails
    # Legacy properties for backward compatibility
    model_name: str = "all-MiniLM-L6-v2"  # Default embedding model
    index_type: IndexType = IndexType.FLAT  # Default index type
    dimension: int = 384  # Embedding dimension, depends on the model
    max_index_size: int = 10000  # Maximum number of vectors in index
    nprobe: int = 1  # Number of clusters to visit for IVF index
    ef_construction: int = 200  # HNSW index parameter
    ef_search: int = 50  # HNSW search parameter
    m: int = 16  # HNSW connections per node


class MemoryItem(BaseModel):
    """Item stored in semantic memory."""
    
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: float = Field(default_factory=time.time)


class SemanticMemory:
    """Semantic memory implementation using FAISS."""
    
    def __init__(self, config: Optional[SemanticMemoryConfig] = None,
                 memory: Optional[Memory] = None):
        """Initialize semantic memory.
        
        Args:
            config: Configuration for semantic memory
            memory: Memory instance for fallback and persistence
        """
        self.config = config or SemanticMemoryConfig()
        self.memory = memory
        self.logger = logging.getLogger("semantic_memory")
        
        # Initialize embedding model
        self.embedding_model = None
        self.index = None
        self.last_save_time = time.time()
        self.items: Dict[str, MemoryItem] = {}
        
        # Statistics
        self.stats = {
            "vector_searches": 0,
            "keyword_searches": 0,
            "vector_stores": 0,
            "keyword_stores": 0,
            "embedding_failures": 0,
            "total_items": 0
        }
        
        # Initialize the system if possible
        self._initialize_embedding_model()
        self._initialize_index()
        
    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            self.logger.warning(
                "SentenceTransformers not available. Semantic search disabled."
            )
            return
            
        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding.model_name
            )
            self.logger.info(
                f"Initialized embedding model: {self.config.embedding.model_name}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
            
    def _initialize_index(self) -> None:
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available. Vector storage disabled.")
            return
            
        try:
            # Try to load from index path if specified
            index_path = getattr(self.config, 'index_path', None)
            if index_path and os.path.exists(index_path):
                try:
                    self.index = faiss.read_index(index_path)
                    self.logger.info(f"Loaded index from {index_path}")
                    return
                except Exception as e:
                    self.logger.error(f"Failed to load index: {e}")
                
            dim = self.config.embedding.dimension  # Default for most models
            index: Optional[faiss.Index] = None
            
            # Create new index
            index_type = self.config.index.index_type.lower()
            if index_type == "flat":
                # Flat index (exact search)
                index = faiss.IndexFlatL2(dim)
            elif index_type == "ivf":
                # Need vectors to train quantizer
                try:
                    # Try to create IVF index
                    nlist = 100  # Number of cells (clusters)
                    if hasattr(self.config.index, "nlist"):
                        nlist = self.config.index.nlist
                        
                    # Create the index
                    metric_type = faiss.METRIC_L2
                    if hasattr(self.config.index, "metric_type") and self.config.index.metric_type != "L2":
                        metric_type = faiss.METRIC_INNER_PRODUCT
                        
                    index = faiss.index_factory(dim, f"IVF{nlist},Flat", metric_type)
                    
                    # Set search parameters
                    if index is not None and hasattr(index, "nprobe"):
                        index.nprobe = min(10, nlist)
                except Exception as e:
                    self.logger.warning(f"Failed to create IVF index: {e}, falling back to Flat")
                    index = faiss.IndexFlatL2(dim)
            elif index_type == "hnsw":
                try:
                    # Hierarchical Navigable Small World - faster with good accuracy
                    m = self.config.index.m
                    factory_str = f"HNSW{m}"
                    index = faiss.index_factory(dim, factory_str)
                    
                    # Set HNSW parameters if the index was created successfully
                    if index is not None and hasattr(index, 'hnsw'):
                        index.hnsw.efConstruction = self.config.index.ef_construction 
                        index.hnsw.efSearch = self.config.index.ef_search
                except Exception as e:
                    self.logger.error(f"Failed to create HNSW index: {e}")
                    # Fall back to flat index
                    index = faiss.IndexFlatL2(dim)
            else:
                self.logger.warning(
                    f"Unknown index type: {self.config.index.index_type}. Using Flat."
                )
                index = faiss.IndexFlatL2(dim)
            
            # Assign the created index to self.index
            self.index = index
            self.logger.info(f"Initialized {index_type} index with dimension {dim}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None
            
    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create embedding for text.
        
        Args:
            text: Text to create embedding for
            
        Returns:
            Embedding vector or None if failed
        """
        # Check prerequisites
        if not text or not isinstance(text, str):
            self.logger.warning("Invalid text provided for embedding")
            return None
            
        if self.embedding_model is None:
            self.logger.warning("Embedding model not initialized")
            return None
        
        # Attempt to create embedding    
        embedding: Optional[np.ndarray] = None
        try:
            # Create embedding - this could theoretically return None in some cases
            embedding = self.embedding_model.encode([text])[0]
            
            # Add a runtime check for safety - mypy sees this as unreachable
            # but we keep it for robustness
            if embedding is None:  # type: ignore
                self.logger.warning("Got None embedding from model")
                return None
            
            # At this point we have a valid embedding
            # Check if normalization is enabled
            normalize_embedding = False
            if hasattr(self.config, 'embedding'):
                if hasattr(self.config.embedding, 'normalize'):
                    normalize_embedding = bool(self.config.embedding.normalize)
            
            # Normalize the embedding if requested
            if normalize_embedding:
                try:
                    norm = float(np.linalg.norm(embedding))
                    if norm > 0.0:
                        embedding = embedding / norm
                    else:
                        self.logger.warning("Embedding has zero norm, skipping normalization")
                except Exception as e:
                    self.logger.warning(f"Failed to normalize embedding: {e}")
                    
            if embedding is not None:
                self.stats["embeddings_created"] += 1
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {e}")
            self.stats["embedding_failures"] += 1
            return None
            
    async def store(self, id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store item in semantic memory.
        
        Args:
            id: Unique identifier for the item
            text: Text content
            metadata: Optional metadata
            
        Returns:
            True if stored successfully
        """
        item = MemoryItem(
            id=id,
            text=text,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Attempt to create and store embedding
        embedding = None
        vector_storage_success = False
        
        # First create the embedding if possible
        embedding = self._create_embedding(text)
        
        # Storage flags
        embedding_created = embedding is not None
        index_available = self.index is not None
        
        # If we have both pieces, attempt vector storage
        if embedding_created and index_available:
            # We'll track if storage succeeds
            vector_storage_success = False
            
            try:
                # Safety check that embedding is not None
                if embedding is None:
                    self.logger.error("Embedding became None unexpectedly")
                    raise ValueError("Cannot convert None embedding to float32")
                    
                # Convert to the format FAISS expects
                embedding_float32 = embedding.astype(np.float32).reshape(1, -1)
                
                # Check index again (to satisfy mypy)
                if self.index is not None:
                    # Add to index
                    self.index.add(embedding_float32)
                    
                    # Store embedding in item for later retrieval
                    item.embedding = embedding.tolist()
                    
                    # Update success flag and stats
                    vector_storage_success = True
                    self.stats["vector_stores"] += 1
            except Exception as e:
                self.logger.error(f"Failed to store in FAISS: {e}")
            
        # Save item metadata
        self.items[id] = item
        self.stats["total_items"] = len(self.items)
        
        # Persist to traditional memory if available
        if self.memory:
            try:
                json_item = item.model_dump_json()
                # Cast to string type since JSON might not be available in MemoryType
                memory_type = cast(str, "json")
                await self.memory.set(f"semantic:{id}", json_item, type=memory_type)
                self.stats["keyword_stores"] += 1
            except Exception as e:
                self.logger.error(f"Failed to persist to Memory: {e}")
                if not vector_storage_success:
                    return False
                
                    
        # Consider saving index if enough time has passed
        if (time.time() - self.last_save_time) > self.config.save_interval:
            self._save_index()
            
        return True
        
    async def search(self, query: str, k: int = 5, 
                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search semantic memory.
        
        Args:
            query: Query text
            k: Number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of matching items with scores
        """
        results: List[Dict[str, Any]] = []
        vector_search_succeeded = False
        
        # Check if we can do vector search - split conditions to help mypy
        model_available = self.embedding_model is not None
        index_available = self.index is not None
        items_available = len(self.items) > 0
        can_vector_search = model_available and index_available and items_available
        
        # Try vector search first if possible
        if can_vector_search:
            self.logger.debug(f"Attempting vector search for query: {query}")
            try:
                # Step 1: Create embedding for query
                query_embedding = self._create_embedding(query)
                
                # Step 2: If we have a valid embedding, search the index
                if query_embedding is not None and self.index is not None:
                    # Convert to numpy array
                    query_array = np.array([query_embedding], dtype=np.float32)
                    
                    # Search the index
                    max_results = min(k, len(self.items))
                    distances, indices = self.index.search(query_array, max_results)
                    
                    # Process search results
                    item_ids = list(self.items.keys())
                    
                    # Iterate through results
                    for i in range(len(indices[0])):
                        idx = indices[0][i]
                        dist = distances[0][i]
                        
                        # Skip invalid indices
                        if idx < 0 or idx >= len(item_ids):
                            continue
                            
                        # Get the item
                        item_id = item_ids[idx]
                        item = self.items[item_id]
                        
                        # Apply metadata filter if provided
                        if metadata_filter:
                            skip = False
                            for key, value in metadata_filter.items():
                                if key not in item.metadata or item.metadata[key] != value:
                                    skip = True
                                    break
                            if skip:
                                continue
                        
                        # Add to results
                        results.append({
                            "id": item.id,
                            "text": item.text,
                            "metadata": item.metadata,
                            "score": float(1.0 / (1.0 + dist)),  # Convert distance to score
                            "match_type": "vector"
                        })
                    
                    # Mark as successful if we processed results
                    vector_search_succeeded = True
                    self.stats["vector_searches"] += 1
            except Exception as e:
                self.logger.error(f"Vector search failed: {e}")
                # We'll fall back to keyword search
                
        # Fallback to keyword search if vector search failed or not available
        should_fallback = not vector_search_succeeded  # First condition
        fallback_enabled = False
        if hasattr(self.config, 'fallback_to_keyword'):
            fallback_enabled = bool(self.config.fallback_to_keyword)
            
        if should_fallback and fallback_enabled:
            self.logger.info("Falling back to keyword search")
            
            # Simple keyword matching
            query_lower = query.lower()
            keyword_results: List[Dict[str, Any]] = []
            
            for item in self.items.values():
                # Check if query appears in text
                if query_lower in item.text.lower():
                    # Apply metadata filter
                    if metadata_filter:
                        match = True
                        for key, value in metadata_filter.items():
                            if key not in item.metadata or item.metadata[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    # Simple scoring based on position of match
                    position = item.text.lower().find(query_lower)
                    score = 1.0 - (float(position) / float(len(item.text))) if len(item.text) > 0 else 0.0
                    
                    keyword_results.append({
                        "id": item.id,
                        "text": item.text,
                        "metadata": item.metadata,
                        "score": score,
                        "match_type": "keyword"
                    })
            
            # Sort by score and limit to k
            # Using proper type annotation for sort key function
            def get_score(x: Dict[str, Any]) -> float:
                return float(x["score"])
            
            keyword_results.sort(key=get_score, reverse=True)
            results.extend(keyword_results[:k - len(results)])
            
            self.stats["keyword_searches"] += 1
            
        # Sort final results by score
        results.sort(key=get_score, reverse=True)
        return results[:k]
        
    def _save_index(self) -> bool:
        """Save FAISS index to disk.
        
        Returns:
            True if saved successfully
        """
        if not self.index:
            self.logger.warning("No index to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            # Save index
            index_path = os.path.join(self.config.cache_dir, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # Save items
            items_path = os.path.join(self.config.cache_dir, "items.json")
            with open(items_path, "w") as f:
                # Convert model objects to dictionaries for serialization
                items_dict = {id: item.model_dump() for id, item in self.items.items()}
                json.dump(items_dict, f)
                
            self.last_save_time = time.time()
            self.logger.info(f"Saved semantic index with {len(self.items)} items")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
            
    def _load_index(self) -> bool:
        """Load FAISS index from disk.
        
        Returns:
            True if loaded successfully
        """
        # Early return conditions
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, cannot load index")
            return False
        
        # Get cache directory path safely
        cache_dir = "./semantic_cache"  # Default value
        if hasattr(self.config, 'cache_dir'):
            cache_dir = self.config.cache_dir
        
        # Define expected file paths
        index_path = os.path.join(cache_dir, "faiss_index.bin")
        items_path = os.path.join(cache_dir, "items.json")
        
        # Verify both files exist
        files_exist = os.path.exists(index_path) and os.path.exists(items_path)
        if not files_exist:
            missing = []
            if not os.path.exists(index_path):
                missing.append("index file")
            if not os.path.exists(items_path):
                missing.append("items file")
            self.logger.warning(f"Cannot load index - missing: {', '.join(missing)}")
            return False
            
        # All preconditions met, attempt to load
        success = False
        try:
            # Load the FAISS index
            loaded_index = faiss.read_index(index_path)
            
            # Load the items dictionary
            with open(items_path, "r") as f:
                items_json = f.read()
                items_dict = json.loads(items_json)
                loaded_items = {}
                for id_str, item_data in items_dict.items():
                    loaded_items[id_str] = MemoryItem(**item_data)
            
            # If we got here, both loads succeeded - update instance
            self.index = loaded_index
            self.items = loaded_items
            self.stats["total_items"] = len(loaded_items)
            
            # Log success
            self.logger.info(f"Loaded semantic index with {len(loaded_items)} items")
            success = True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            success = False
            
        return success
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about semantic memory.
        
        Returns:
            Dictionary of statistics
        """
        # Add some additional calculated stats
        vector_ratio: float = 0.0
        total_searches = self.stats["vector_searches"] + self.stats["keyword_searches"]
        if total_searches > 0:
            vector_ratio = float(self.stats["vector_searches"]) / float(total_searches)
            
        stats = {
            **self.stats,
            "vector_search_ratio": vector_ratio,
            "embedding_model": self.embedding_model is not None,
            "index_available": self.index is not None,
            "index_type": self.config.index.index_type if self.index is not None else None,
        }
        
        return stats
