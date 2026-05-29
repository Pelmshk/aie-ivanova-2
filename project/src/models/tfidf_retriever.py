"""TF-IDF + Cosine Similarity."""
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_retriever import BaseRetriever

class TfidfRetriever(BaseRetriever):
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df,
            lowercase=True, strip_accents='unicode'
        )
        self.doc_matrix = None
        self.doc_ids = None
    
    def fit(self, documents: List[str], doc_ids: Optional[List] = None) -> None:
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        self.doc_ids = doc_ids or list(range(len(documents)))
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]