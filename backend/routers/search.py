import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from services.auth import get_current_active_user
from services.embedding import embedder
from services.vector_db import vector_db

from models.auth import User
from models.request_models import SearchRequest, SearchResponse, SearchResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest, current_user: User = Depends(get_current_active_user)
):
    # Извлекаем эмбеддинг для запроса
    subject_embedding = embedder.get_embeddings([request.subject])[0]
    description_embedding = embedder.get_embeddings([request.description])[0]

    # Объединяем эмбеддинги
    query_embedding = np.concatenate([subject_embedding, description_embedding])

    # Ищем похожие векторы
    search_results = vector_db.search_vectors(query_embedding, limit=request.limit, collection_name=request.collection)

    # Форматируем результаты
    results = [
        SearchResult(
            id=result["id"],
            request_id=result["request_id"],  # Добавляем это поле
            subject=result["subject"],
            description=result["description"],
            class_name=result["class_name"],
            score=result["score"],
        )
        for result in search_results
    ]

    return SearchResponse(results=results)
