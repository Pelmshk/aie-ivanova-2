FROM python:3.11-slim

# ===== Метаданные =====
LABEL description="Recipe Search API with BM25 retriever"
LABEL version="2.0"

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/artifacts/models/bm25.pkl \
    DATA_PATH=/app/data/processed/recipes_processed.parquet \
    PORT=8000

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей и установка
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода и артефактов
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/processed/ ./data/processed/

# Создание __init__.py при его отсутствии
RUN touch src/__init__.py && \
    touch src/service/__init__.py && \
    touch src/models/__init__.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')" || exit 1

# Экспорт порта
EXPOSE ${PORT}

# Команда запуска
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]