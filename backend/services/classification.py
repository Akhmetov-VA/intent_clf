import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings
from services.vector_db import vector_db

logger = logging.getLogger("classification_service")


class FilteredKNNClassificationService:
    """Two-stage classifier using KNN search in Qdrant and reranker filtering."""

    def __init__(self) -> None:
        self.k = settings.KNN_NEIGHBORS
        self.threshold = settings.RERANK_THRESHOLD
        self.device = settings.DEVICE

        logger.info(
            "Loading reranker model %s on %s", settings.RERANKER_MODEL, self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(settings.RERANKER_MODEL)
            .to(self.device)
        )
        self.model.eval()

    def _rerank_scores(self, query: str, docs: List[str]) -> List[float]:
        """Computes reranker scores for a query and list of documents."""
        pairs = [(query, doc) for doc in docs]
        encoded = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            max_length=settings.MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits.cpu().numpy()

        if logits.shape[1] == 2:
            exp = np.exp(logits)
            probs = exp[:, 1] / exp.sum(axis=1)
            return probs.tolist()
        return logits[:, 0].tolist()

    def predict(
        self, subject: str, description: str, query_vector: np.ndarray, collection_name: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """Return class probabilities for the provided query."""
        query_text = f"{subject} {description}"

        # Search in vector database
        neighbors = vector_db.search_vectors(query_vector, limit=self.k, collection_name=collection_name)

        if not neighbors:
            logger.warning("No neighbors found in vector DB")
            return []

        neighbor_texts = [n.get("subject", "") + " " + n.get("description", "") for n in neighbors]
        neighbor_labels = [n.get("class_name", "") for n in neighbors]
        sims = [float(n.get("score", 0.0)) for n in neighbors]

        scores = self._rerank_scores(query_text, neighbor_texts)
        kept_indices = [i for i, s in enumerate(scores) if s >= self.threshold]

        def _aggregate(indices: List[int]) -> List[Dict[str, float]]:
            class_scores: Dict[str, float] = {}
            total_sim = 0.0
            for i in indices:
                lbl = neighbor_labels[i]
                sim = sims[i]
                class_scores[lbl] = class_scores.get(lbl, 0.0) + sim
                total_sim += sim

            if total_sim <= 1e-12:
                counts = Counter([neighbor_labels[i] for i in indices])
                total = sum(counts.values())
                return [
                    {"class_name": lbl, "probability": cnt / total}
                    for lbl, cnt in counts.most_common()
                ]

            return [
                {"class_name": lbl, "probability": score / total_sim}
                for lbl, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            ]

        if kept_indices:
            predictions = _aggregate(kept_indices)
        else:
            # fallback to using all neighbors
            predictions = _aggregate(list(range(len(neighbors))))

        return predictions


# Singleton instance
classifier = FilteredKNNClassificationService()
