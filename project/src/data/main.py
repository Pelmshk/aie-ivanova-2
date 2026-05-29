import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.data.loader import load_processed_recipes
from src.models.coverage_search import CoverageSearch
from src.service.schemas import RecipeRequest, RecipeResponse, HealthResponse

app = FastAPI(title="Recipe Recommender API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

logger = logging.getLogger(__name__)
model = None
df_recipes = None

@app.on_event("startup")
async def startup():
    global model, df_recipes
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed_recipes.csv"
    logger.info(f"Загрузка данных: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"Файл не найден: {csv_path}")
        raise FileNotFoundError(f"Запустите 01_eda.ipynb для создания файла")
    
    df_recipes = load_processed_recipes(str(csv_path))
    logger.info("Инициализация CoverageSearch...")
    model = CoverageSearch(df_recipes)
    logger.info(f"Сервис готов. Загружено {len(df_recipes)} рецептов.")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model else "starting",
        model_version="v1",
        loaded_recipes=len(df_recipes) if df_recipes is not None else 0
    )

@app.post("/predict", response_model=list[RecipeResponse])
async def predict(req: RecipeRequest):
    start = time.time()
    try:
        results = model.search(req.ingredients, top_k=req.max_results)
        responses = []
        user_set = set(ing.lower().strip() for ing in req.ingredients)
        for r in results:
            matched = list(user_set & set(r['ingredients']))
            responses.append(RecipeResponse(
                title=r['title'], similarity_score=r['similarity_score'],
                difficulty=r.get('difficulty', 'medium'),
                cooking_time=r.get('cooking_time', 'medium'),
                ingredients=r['ingredients'], instructions=r['instructions'],
                matched_ingredients=matched
            ))
        logger.info(f"Ответов: {len(responses)} | ⏱ {time.time()-start:.3f}s")
        return responses
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")