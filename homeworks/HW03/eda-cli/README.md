# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`);

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports_example --max-hist-columns 10 --top-k-categories 2 --title "Report" --min-missing-share 0.01
```
Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`);
- `--out_dir` – каталог для сохнания отчёта (по умолчанию `reports`);
- `--max-hist-columns` – сколько числовых колонок включать в набор гистограмм (по умолчанию `6`);
- `--top-k-categories` – сколько top-значений выводить для категориальных признаков (по умолчанию `5`);
- `--title` – (по умолчанию `# EDA-отчёт\n\n`);
- `--min-missing-share` – (по умолчанию `0.3`).

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```

## По ДЗ сделано:

- Добавлен флаг `has_constant_columns`