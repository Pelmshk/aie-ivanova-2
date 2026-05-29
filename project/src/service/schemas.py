from pydantic import BaseModel, Field
from typing import List, Optional

class RecipeRequest(BaseModel):
    ingredients: List[str] = Field(..., min_items=1, description="Список ингредиентов")
    max_results: int = Field(default=10, ge=1, le=50)

class RecipeResponse(BaseModel):
    title: str
    similarity_score: float
    ingredients: List[str]
    instructions: str
    matched_count: int  # Сколько именно ингредиентов из запроса совпало

class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_recipes: int