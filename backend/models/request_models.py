from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RequestItem(BaseModel):
    id: str = Field(..., description="Идентификатор заявки")
    subject: str = Field(..., description="Тема обращения")
    description: str = Field(..., description="Описание обращения")
    class_name: Optional[str] = Field(
        None, description="Класс обращения (если известен)"
    )
    collection: Optional[str] = Field(None, description="Имя коллекции Qdrant")
    task: Optional[str] = Field(None, description="Задача обращения (если известна)")

    class Config:
        schema_extra = {
            "example": {
                "id": "INC0027099",
                "subject": "Старая версия 1С клиента. Садовники д.4 к.2",
                "description": "Удалить старые версии 1С клиента на пк коменданта.",
                "class_name": "Сопровождение сервисов сотрудника",
                "task": "1С клиент",
            }
        }


class BatchRequestItem(BaseModel):
    items: List[RequestItem] = Field(..., description="Список обращений для обработки")
    collection: Optional[str] = Field(None, description="Имя коллекции Qdrant для загрузки")

    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "subject": "Проблема с доступом к системе",
                        "description": "Не могу войти в личный кабинет после обновления пароля",
                    },
                    {
                        "subject": "Ошибка при оплате",
                        "description": "При попытке оплаты возникает ошибка сервера",
                    },
                ]
            }
        }


class PredictionResult(BaseModel):
    class_name: str = Field(..., description="Предсказанный класс")
    probability: float = Field(..., description="Вероятность класса")


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult] = Field(
        ..., description="Список предсказаний с вероятностями"
    )

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"class_name": "Технические проблемы", "probability": 0.85},
                    {"class_name": "Финансовые вопросы", "probability": 0.10},
                    {"class_name": "Другое", "probability": 0.05},
                ]
            }
        }


class UploadResponse(BaseModel):
    success: bool = Field(..., description="Статус операции")
    message: str = Field(..., description="Сообщение о результате")
    ids: Optional[List[str]] = Field(
        None, description="Идентификаторы загруженных документов"
    )

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Данные успешно загружены",
                "ids": ["1", "2", "3"],
            }
        }


class SearchRequest(BaseModel):
    subject: str = Field("", description="Тема для поиска")
    description: str = Field("", description="Описание для поиска")
    limit: int = Field(10, description="Количество результатов")
    collection: Optional[str] = Field(None, description="Имя коллекции Qdrant для поиска")


class SearchResult(BaseModel):
    id: str = Field(..., description="Идентификатор документа в Qdrant")
    request_id: str = Field(..., description="Идентификатор исходной заявки")
    subject: str = Field(..., description="Тема обращения")
    description: str = Field(..., description="Описание обращения")
    class_name: str = Field(..., description="Класс обращения")
    score: float = Field(..., description="Оценка релевантности")


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Результаты поиска")

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "1",
                        "subject": "Проблема с доступом к системе",
                        "description": "Не могу войти в личный кабинет после обновления пароля",
                        "class_name": "Технические проблемы",
                        "score": 0.92,
                    }
                ]
            }
        }

class CollectionRequest(BaseModel):
    collection: Optional[str] = Field(None, description="Имя коллекции Qdrant")

