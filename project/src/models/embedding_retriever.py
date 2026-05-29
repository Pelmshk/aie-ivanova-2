"""Поиск на основе эмбеддингов (Sentence-Transformers)."""
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
from .base_retriever import BaseRetriever

class EmbeddingRetriever(BaseRetriever):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_embeddings = None
        self.doc_ids = None
    
    def fit(self, documents: List[str], doc_ids: Optional[List] = None) -> None:
        self.doc_embeddings = self.model.encode(documents, convert_to_tensor=True, device=self.device)
        self.doc_ids = doc_ids or list(range(len(documents)))
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple]:
        query_emb = self.model.encode(query, convert_to_tensor=True, device=self.device)
        cos_scores = util.cos_sim(query_emb, self.doc_embeddings)[0].cpu().numpy()
        top_indices = cos_scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(cos_scores[i])) for i in top_indices if cos_scores[i] > 0.1]