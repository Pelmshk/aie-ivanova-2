"""BM25 (классический поисковый алгоритм)."""
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from .base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = None
        self.tokenized_docs = None
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()
    
    def fit(self, documents: List[str], doc_ids: Optional[List] = None) -> None:
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        self.doc_ids = doc_ids or list(range(len(documents)))
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]