from config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, health, predict, search, upload
from services.vector_db import vector_db

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API для классификации входящих обращений",
)


# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Замените на список разрешенных источников в продакшене
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(upload.router)
app.include_router(search.router)


@app.on_event("startup")
async def startup_event():
    vector_db.init_collection(vector_db.collection_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
