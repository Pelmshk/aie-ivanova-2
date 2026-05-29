"""Базовый интерфейс для всех моделей поиска рецептов."""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path

class BaseRetriever(ABC):
    """Абстрактный класс для моделей поиска."""
    
    @abstractmethod
    def fit(self, documents: List[str], doc_ids: Optional[List] = None) -> None:
        """Обучение/индексация документов."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple]:
        """Поиск top_k документов по запросу. Возвращает [(doc_id, score), ...]."""
        pass
    
    def save(self, path: str) -> None:
        """Сохранение модели."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseRetriever':
        """Загрузка модели."""
        with open(path, 'rb') as f:
            return pickle.load(f)