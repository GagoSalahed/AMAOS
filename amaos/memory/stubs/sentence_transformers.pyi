"""Type stubs for sentence_transformers library"""
from typing import List, Union, Dict, Any, Optional
import numpy as np

class SentenceTransformer:
    """SentenceTransformer class for encoding text to embeddings."""
    def __init__(self, model_name_or_path: str): ...
    
    def encode(self, 
               sentences: Union[str, List[str]], 
               batch_size: int = 32, 
               show_progress_bar: bool = False,
               normalize_embeddings: bool = False) -> np.ndarray: ...
    
    def get_max_seq_length(self) -> int: ...
    def get_sentence_embedding_dimension(self) -> int: ...
