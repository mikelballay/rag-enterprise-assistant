"""
Cross-encoder reranker for the RAG retrieval pipeline.

The model (cross-encoder/ms-marco-MiniLM-L-6-v2) is ~85 MB and is loaded
lazily on the first call to ``rerank``, so import time stays instant.

Usage
-----
    from app.services.reranker import get_reranker

    reranker = get_reranker()           # returns the module-level singleton
    top_docs = reranker.rerank(query, docs, top_n=3)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Wraps a sentence-transformers CrossEncoder model.

    The underlying model is not loaded until the first call to ``rerank``,
    so creating a ``Reranker`` instance has zero I/O cost.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading reranker model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Reranker model loaded.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        """
        Score every (query, document) pair and return the *top_n* highest-
        scoring documents in descending relevance order.

        Parameters
        ----------
        query : str
            The user's question.
        documents : list[str]
            Candidate document texts to rank.
        top_n : int
            How many documents to keep.

        Returns
        -------
        list[str]
            The *top_n* most relevant documents (highest score first).
        """
        if not documents:
            return []

        top_n = min(top_n, len(documents))

        self._ensure_loaded()

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Module-level singleton — one model load per process
# ---------------------------------------------------------------------------

_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """Return the shared ``Reranker`` instance, creating it if needed."""
    global _instance
    if _instance is None:
        _instance = Reranker()
    return _instance
