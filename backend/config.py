import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Classification API"
    APP_VERSION: str = "1.0.0"

    # Настройки безопасности
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 360

    # Настройки модели
    MODEL_NAME: str = "intfloat/multilingual-e5-large"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 32
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"
    USE_CUDA: str = os.getenv("USE_CUDA", "0")  # Default to "0" (CPU) if not specified

    # Настройки Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "requests")

    # Настройки классификации
    KNN_NEIGHBORS: int = 5
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    RERANK_THRESHOLD: float = 0.2

    class Config:
        env_file = ".env"


settings = Settings()
