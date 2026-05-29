"""Сервис поиска рецептов по ингредиентам."""
import logging
from pathlib import Path
from typing import List, Dict
from src.models.bm25_retriever import BM25Retriever
import pandas as pd

logger = logging.getLogger(__name__)

class RecipeSearchService:
    """Сервис для поиска рецептов по списку ингредиентов."""
    
    def __init__(self, model_path: str, data_path: str):
        """
        Инициализация сервиса.
        
        Args:
            model_path: Путь к сохранённой модели BM25
            data_path: Путь к обработанным данным рецептов
        """
        logger.info(f"Загрузка модели из {model_path}")
        self.retriever = BM25Retriever.load(model_path)
        
        logger.info(f"Загрузка данных из {data_path}")
        self.recipes_df = pd.read_parquet(data_path)
        
        logger.info(f"Загружено {len(self.recipes_df)} рецептов")
    
    def search_by_ingredients(
        self, 
        ingredients: List[str], 
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Поиск рецептов по списку ингредиентов.
        
        Args:
            ingredients: Список ингредиентов для поиска
            top_k: Количество лучших результатов
            min_score: Минимальный порог схожести
            
        Returns:
            Список словарей с найденными рецептами
        """
        if not ingredients:
            logger.warning("Пустой список ингредиентов")
            return []
        
        # Формируем поисковый запрос
        query = " ".join(ingredients)
        logger.info(f"Поиск по запросу: {query}")
        
        # Ищем похожие рецепты
        results = self.retriever.search(query, top_k=top_k * 2)
        
        # Фильтруем по порогу и enrich данными
        found_recipes = []
        for doc_id, score in results:
            if score < min_score:
                continue
                
            recipe = self.recipes_df.iloc[doc_id]
            
            found_recipes.append({
                'recipe_name': recipe.get('recipe_name', 'Unknown'),
                'similarity_score': round(score, 3),
                'ingredients': set(recipe.get('ingredients_clean', [])),
                'cuisine': recipe.get('cuisine_path', ''),
                'total_time': recipe.get('total_time'),
                'rating': recipe.get('rating'),
                'url': recipe.get('url', ''),
            })
        
        # Сортируем по score и берём top_k
        found_recipes.sort(key=lambda x: x['similarity_score'], reverse=True)
        found_recipes = found_recipes[:top_k]
        
        logger.info(f"Найдено {len(found_recipes)} рецептов")
        return found_recipes