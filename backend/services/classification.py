from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from config import settings
from services.vector_db import vector_db


class ClassificationService:
    def __init__(self):
        self.k = settings.KNN_NEIGHBORS

    def predict(self, query_vector: np.ndarray) -> List[Dict[str, float]]:
        """
        Предсказывает класс на основе K ближайших соседей
        """
        # Получаем K ближайших соседей из Qdrant
        neighbors = vector_db.search_vectors(query_vector, limit=self.k)

        # Извлекаем классы соседей
        neighbor_classes = [neighbor["class_name"] for neighbor in neighbors]

        # Считаем частоту каждого класса
        class_counts = Counter(neighbor_classes)
        total = sum(class_counts.values())

        # Вычисляем вероятности классов
        predictions = [
            {"class_name": class_name, "probability": count / total}
            for class_name, count in class_counts.most_common()
        ]

        return predictions


# Создаем синглтон для классификации
classifier = ClassificationService()
