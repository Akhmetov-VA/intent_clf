from fastapi import APIRouter, Depends, HTTPException
from services.auth import get_current_active_user
from services.classification import classifier
from services.embedding import embedder

from models.auth import User
from models.request_models import PredictionResponse, RequestItem

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: RequestItem, current_user: User = Depends(get_current_active_user)
):
    # Извлекаем эмбединги
    combined_embedding = embedder.get_combined_embeddings(
        [request.subject], [request.description]
    )

    # Получаем предсказания с учетом текстов для двухуровневого классификатора
    predictions = classifier.predict(
        request.subject,
        request.description,
        combined_embedding[0],
    )

    return PredictionResponse(predictions=predictions)
