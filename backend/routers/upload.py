import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from services.auth import get_current_active_user
from services.embedding import embedder
from services.vector_db import vector_db

from models.auth import User
from models.request_models import (
    BatchRequestItem,
    UploadResponse,
    CollectionRequest,
    CopyCollectionRequest,
)

# Добавляем логгер
logger = logging.getLogger("upload_router")

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_data(
    request: BatchRequestItem, current_user: User = Depends(get_current_active_user)
):
    subjects = [item.subject for item in request.items]
    descriptions = [item.description for item in request.items]

    # Извлекаем эмбединги
    combined_embeddings = embedder.get_combined_embeddings(subjects, descriptions)

    # Подготавливаем payload для Qdrant
    payloads = []
    for i, item in enumerate(request.items):
        # Проверяем наличие id в запросе
        if not hasattr(item, "id") or item.id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required 'id' field for item at index {i}",
            )

        payload = {
            "request_id": item.id,  # Используем id из запроса как request_id
            "subject": item.subject,
            "description": item.description,
            "class_name": getattr(item, "class_name", "unknown"),
            "task": getattr(item, "task", "unknown"),  # Добавляем поле task
        }
        payloads.append(payload)

    # Загружаем векторы в Qdrant
    try:
        ids = vector_db.upload_vectors(combined_embeddings, payloads, collection_name=request.collection)
        return UploadResponse(success=True, message="Данные успешно загружены", ids=ids)
    except Exception as e:
        logger.error(f"Error uploading vectors: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading vectors: {str(e)}"
        )

@router.post("/clear_index", response_model=UploadResponse)
async def clear_index(request: CollectionRequest = None, current_user: User = Depends(get_current_active_user)):
    try:
        vector_db.clear_collection(collection_name=request.collection if request else None)
        return UploadResponse(success=True, message="Индекс успешно очищен", ids=[])
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing index: {str(e)}"
        )


@router.post("/create_collection", response_model=UploadResponse)
async def create_collection(
    request: CollectionRequest, current_user: User = Depends(get_current_active_user)
):
    try:
        vector_db.init_collection(request.collection)
        return UploadResponse(success=True, message="Коллекция создана", ids=[])
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@router.post("/copy_collection", response_model=UploadResponse)
async def copy_collection(
    request: CopyCollectionRequest, current_user: User = Depends(get_current_active_user)
):
    try:
        vector_db.copy_collection(request.source, request.dest)
        return UploadResponse(success=True, message="Коллекция скопирована", ids=[])
    except Exception as e:
        logger.error(f"Error copying collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error copying collection: {str(e)}")
