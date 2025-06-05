import os

# Конфигурационные параметры
API_URL = os.getenv("API_URL", "http://localhost:8190") 
USERNAME = os.getenv("USERNAME", "admin") 
PASSWORD = os.getenv("PASSWORD", "secret")

# Имя коллекции по умолчанию для тестового стенда
TEST_COLLECTION = os.getenv("TEST_COLLECTION", "test_requests")

# Примеры запросов для выбора
DEFAULT_EXAMPLES = [
    {
        "id": "2102_0",
        "description": "'Application not completed, Login problem'",
        "subject": None,
        "class": "Application Related/Login Issue",
        "task": "",
    }
    # Можно добавить больше примеров в будущем
]
