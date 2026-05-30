import logging
import sys
from pathlib import Path
from typing import List, Optional
import time
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parents[1]))

from src.service.recipe_search_service import RecipeSearchService
from .middleware import ObservabilityMiddleware, metrics
from configs.settings import settings

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Recipe Search API",
    description="API для поиска рецептов по ингредиентам",
    version="2.0.0"
)

app.add_middleware(ObservabilityMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для сервиса
search_service: Optional[RecipeSearchService] = None


class IngredientSearchRequest(BaseModel):
    """Запрос на поиск рецептов по ингредиентам"""
    ingredients: List[str] = Field(
        ..., 
        description="Список ингредиентов",
        example=["chicken", "garlic", "olive oil"]
    )
    top_k: int = Field(
        default=10, 
        ge=1, 
        le=50,
        description="Количество результатов"
    )
    min_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Минимальный порог схожести"
    )


class RecipeResponse(BaseModel):
    """Информация о найденном рецепте"""
    recipe_name: str
    similarity_score: float
    ingredients: List[str]
    cuisine: Optional[str] = None
    total_time: Optional[str] = None
    rating: Optional[float] = None
    url: Optional[str] = None


class SearchResponse(BaseModel):
    """Ответ API с результатами поиска"""
    query: List[str]
    results_count: int
    results: List[RecipeResponse]


class HealthResponse(BaseModel):
    """Ответ эндпоинта health check"""
    status: str
    service: str
    recipes_loaded: int


class MetricsResponse(BaseModel):
    """Ответ эндпоинта metrics"""
    uptime_seconds: float
    requests_total: int
    requests_success: int
    requests_error: int
    avg_response_time_seconds: float


@app.on_event("startup")
async def startup_event():
    """Загрузка модели и данных при старте приложения"""
    global search_service
    
    models_dir = Path(__file__).parents[2] / "artifacts" / "models"
    data_dir = Path(__file__).parents[2] / "data" / "processed"
    
    model_path = models_dir / "bm25.pkl"
    data_path = data_dir / "recipes_processed.parquet"
    
    if not model_path.exists():
        raise RuntimeError(f"Модель не найдена: {model_path}")
    if not data_path.exists():
        raise RuntimeError(f"Данные не найдены: {data_path}")
    
    search_service = RecipeSearchService(
        model_path=str(model_path),
        data_path=str(data_path)
    )
    logger.info("Сервис успешно инициализирован")


# Endpoints

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    uptime = time.time() - metrics["start_time"]
    avg_time = (
        sum(metrics["response_times_seconds"]) / len(metrics["response_times_seconds"])
        if metrics["response_times_seconds"] else 0.0
    )
    return MetricsResponse(
        uptime_seconds=round(uptime, 2),
        requests_total=metrics["requests_total"],
        requests_success=metrics["requests_success"],
        requests_error=metrics["requests_error"],
        avg_response_time_seconds=round(avg_time, 4),
    )
    

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if search_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return HealthResponse(
        status="ok",
        service="recipe-search-api",
        recipes_loaded=len(search_service.recipes_df)
    )


@app.post("/predict")
@app.post("/search", response_model=SearchResponse)
async def search_recipes(request: IngredientSearchRequest):
    """
    Поиск рецептов по списку ингредиентов.
    Возвращает рецепты, отсортированные по релевантности.
    """
    if search_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = search_service.search_by_ingredients(
            ingredients=request.ingredients,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        return SearchResponse(
            query=request.ingredients,
            results_count=len(results),
            results=[RecipeResponse(**r) for r in results]
        )
    
    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Корневой endpoint с информацией о сервисе"""
    return {
        "service": "Recipe Search API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "search": "/search (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)