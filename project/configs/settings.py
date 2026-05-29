import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env") # Загрузка конфигурации из .env

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Recipe Search API")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Пути
    BASE_DIR = Path(__file__).parents[2]
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "artifacts/models/bm25.pkl"))
    DATA_PATH: str = os.getenv("DATA_PATH", str(BASE_DIR / "data/processed/recipes_processed.parquet"))
    
    # Параметры API
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "50"))
    MIN_SCORE_THRESHOLD: float = float(os.getenv("MIN_SCORE_THRESHOLD", "0.0"))

settings = Settings()