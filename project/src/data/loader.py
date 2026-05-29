import pandas as pd

def load_processed_recipes(csv_path: str) -> pd.DataFrame:
    """Загружает CSV, уже обработанный в 01_eda.ipynb"""
    df = pd.read_csv(csv_path)
    required = {'Title', 'Cleaned_Ingredients', 'Instructions'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {missing}")
    return df.reset_index(drop=True)