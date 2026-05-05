# semantic-nav-search

`semantic-nav-search` --- FastAPI микросервис для обработки запросов о навигационных данных из CSV. LLM(OpenRouter) используется только для классификации типа запроса и параметров извлечения информации. Фильтрация, Аггрегация, рассчет расстояния, разгон/торможение и геопространственные вычесления выполняются с помощью python.

## Пререквизиты

- Python 3.11
- Docker и Docker Compose, опционально
- API-ключ OpenRouter
- CSV-файлы в `./data/` с нужной навигационной схемой

## Запуск

Через Docker:

```bash
cd semantic-nav-search
cp .env.example .env
docker compose up --build
```

Без Docker (Linux):

```bash
cd semantic-nav-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Без Docker (Windows):

```powershell
cd semantic-nav-search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Измените `.env`, заполнив `OPENROUTER_API_KEY=` перед вызовом `/query`.

## Примеры запросов

Аггрегация:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"покажи максималку"}'
```

Фильтрация:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"где gps глючил"}'
```

Временной срез:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"в сумерках"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"show daytime points"}'
```

Торможение:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"резко тормозил"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"hard braking below minus three"}'
```

Геофильтр:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"на м11"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"points in Moscow"}'
```

## Примеры ответов

Аггрегация:

```json
{
  "status": "success",
  "query": "top speed",
  "intent": "aggregation",
  "result": {
    "metric": "max_speed",
    "total_points": 5230,
    "speed_kmh": 98.42,
    "timestamp": "2023-11-14 18:13:20",
    "latitude": 56.231,
    "longitude": 32.821
  }
}
```

Фильтрация по качеству позиционирования:

```json
{
  "status": "success",
  "query": "где gps глючил",
  "intent": "filter_quality",
  "result": {
    "quality": "poor",
    "total_points": 84,
    "percentage_of_total": 1.61,
    "points": [
      {
        "timestamp": "2023-11-14 17:02:00",
        "latitude": 56.1,
        "longitude": 31.5,
        "pos_type": 19,
        "horizontal_speed": 42.12
      }
    ]
  }
}
```

Временной срез:

```json
{
  "status": "success",
  "query": "в сумерках",
  "intent": "time_slice",
  "result": {
    "period": "twilight",
    "time_range": "16:00-19:00",
    "total_points": 630,
    "points": [
      {
        "timestamp": "2023-11-14 16:05:00",
        "latitude": 56.15,
        "longitude": 31.61,
        "horizontal_speed": 51.3,
        "azimuth": 182.0
      }
    ]
  }
}
```

Торможение:

```json
{
  "status": "success",
  "query": "резко тормозил",
  "intent": "braking",
  "result": {
    "threshold": -2.0,
    "total_braking_events": 2,
    "max_deceleration": -4.2,
    "avg_deceleration": -3.1,
    "events": [
      {
        "start_timestamp": "2023-11-14 18:10:00",
        "end_timestamp": "2023-11-14 18:10:04",
        "latitude": 56.2,
        "longitude": 31.9,
        "max_deceleration": -4.2,
        "speed_before": 76.4,
        "speed_after": 43.8,
        "duration_seconds": 4
      }
    ]
  }
}
```

Геофильтр:

```json
{
  "status": "success",
  "query": "на м11",
  "intent": "geo_filter",
  "result": {
    "region": "м11",
    "bounds": {"lat_min": 55.5, "lat_max": 60.0, "lon_min": 30.0, "lon_max": 37.5},
    "total_points": 3120,
    "entry_time": "2023-11-14 16:00:00",
    "exit_time": "2023-11-14 17:22:00",
    "time_spent_minutes": 82.0,
    "points": [
      {"timestamp": "2023-11-14 16:00:00", "latitude": 56.0, "longitude": 31.0, "speed": 45.6}
    ]
  }
}
```

## Архитектура

Приложение один раз загружает все CSV-файлы при старте в singleton `DataStore`, валидирует обязательную схему и рассчитывает производные поля: скорость, ускорение и московское локальное время. `/query` отправляет текст пользователя в OpenRouter только для классификации intent и извлечения параметров, нормализует JSON от LLM через константы на стороне Python и передаёт запрос в pandas-обработчик для детерминированного расчёта.

## Добавление новых intent

Добавьте новый модуль-обработчик в `app/handlers/`, определите чистую функцию `handle(df, params)`, добавьте intent в `INTENT_MAP` и расширьте нормализацию `ParsedQuery` в `query_parser.py`. Обновите prompt примерами, храните бизнес-константы на стороне Python и добавьте тесты, которые проверяют обработчик без вызова LLM.
