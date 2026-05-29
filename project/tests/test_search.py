def test_search_by_ingredients(client):
    payload = {
        "ingredients": ["chicken", "garlic"],
        "top_k": 3,
        "min_score": 0.0
    }
    resp = client.post("/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) <= 3
    
    # Проверка структуры первого результата
    if data["results"]:
        recipe = data["results"][0]
        assert "recipe_name" in recipe
        assert "ingredients" in recipe
        assert "similarity_score" in recipe

def test_search_empty_ingredients(client):
    payload = {"ingredients": [], "top_k": 5}
    resp = client.post("/search", json=payload)
    assert resp.status_code == 200
    assert resp.json()["results_count"] == 0

def test_search_validation_error(client):
    # top_k больше максимума
    payload = {"ingredients": ["salt"], "top_k": 100}
    resp = client.post("/search", json=payload)
    assert resp.status_code == 422  # FastAPI валидация Pydantic