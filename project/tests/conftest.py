import pytest
from fastapi.testclient import TestClient
from src.service.main import app

@pytest.fixture(scope="module")
def client():
    """Тестовый клиент FastAPI."""
    with TestClient(app) as c:
        yield c