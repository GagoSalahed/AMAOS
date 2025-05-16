"""Type stubs for FAISS library"""
from typing import Any, Tuple, List, Optional
import numpy as np

# Metric types
METRIC_L2: int
METRIC_INNER_PRODUCT: int

class Index:
    """Base class for all FAISS indices."""
    def add(self, x: np.ndarray) -> None: ...
    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...
    def train(self, x: np.ndarray) -> None: ...
    def reset(self) -> None: ...
    
    # HNSW specific attributes
    class HNSW:
        efConstruction: int
        efSearch: int
    
    hnsw: HNSW
    
    # IVF specific attributes
    nprobe: int

class IndexFlatL2(Index):
    """Flat index with L2 distance."""
    def __init__(self, d: int): ...

class IndexHNSWFlat(Index):
    """HNSW index with flat storage."""
    def __init__(self, d: int, M: int, metric: int = METRIC_L2): ...

def index_factory(d: int, description: str, metric: Optional[int] = None) -> Index: ...
def read_index(path: str) -> Index: ...
def write_index(index: Index, path: str) -> None: ...
