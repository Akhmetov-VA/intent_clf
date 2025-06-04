# Intent Classification Service

Проект предоставляет REST API и веб-интерфейс для классификации обращений и поиска похожих документов. В качестве хранилища векторов используется **Qdrant**, а для извлечения эмбеддингов — модель семейства **E5**.

## Возможности

- классификация обращений по теме и описанию;
- поиск похожих документов в базе;
- загрузка новых данных в Qdrant и очистка коллекции;
- веб-интерфейс на Streamlit для работы с сервисом.

## Структура репозитория

- `backend/` — FastAPI приложение с бизнес‑логикой;
- `frontend/` — интерфейс Streamlit;
- `notebooks/` — Jupyter‑ноутбуки и скрипты для экспериментов;
- `docker-compose.yml` — конфигурация для запуска всего стека;
- `pyproject.toml` — зависимости Python.

## Запуск через Docker Compose

```bash
docker compose up --build
```

Поднимутся сервисы:

- **api** — FastAPI на порту `8000`;
- **qdrant** — хранилище векторов на порту `6333`;
- **frontend** — Streamlit приложение на порту `8501`.

## Основные эндпоинты API

- `GET /health` — проверка состояния сервиса;
- `POST /token` — получение JWT‑токена;
- `POST /predict` — классификация одного обращения;
- `POST /search` — поиск похожих документов;
- `POST /upload` — загрузка набора документов в коллекцию;
- `POST /clear_index` — очистка коллекции.

Эндпоинты `/predict`, `/search`, `/upload` и `/clear_index` требуют авторизации.

## Переменные окружения

Ключевые настройки задаются переменными окружения (см. `docker-compose.yml`):

- `SECRET_KEY` — секретный ключ для токенов;
- `QDRANT_HOST`, `QDRANT_PORT` — адрес Qdrant;
- `QDRANT_COLLECTION` — имя коллекции в хранилище;
- `USE_CUDA` — использовать ли GPU (`1` или `0`).

## Локальный запуск

Требуется Python 3.10+. Установка зависимостей и запуск API:

```bash
cd backend
pip install uv
uv sync
uv run uvicorn main:app --reload
```

Запуск фронтенда:

```bash
cd ../frontend
pip install streamlit pandas seaborn matplotlib requests scikit-learn tqdm openpyxl xlrd
streamlit run main.py
```

---

Проект демонстрирует использование FastAPI и Qdrant для классификации обращений и может служить базой для дальнейших разработок.
