from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_PATH: str = str(PROJECT_ROOT / "data" / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_config() -> Settings:
    return Settings()