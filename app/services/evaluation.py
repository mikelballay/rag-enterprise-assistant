"""
RAGAS-based evaluation pipeline for the RAG system.

Compatible with ragas>=0.1.7,<0.2.0  (HuggingFace-datasets-based API).

Usage
-----
    from app.services.evaluation import run_evaluation

    # Against the local Python function (offline / dev):
    results = run_evaluation(test_dataset)

    # Against the live Cloud Run endpoint:
    import httpx
    def cloud_answer(question: str) -> str:
        r = httpx.post("https://<service>/chat", json={"question": question}, timeout=30)
        r.raise_for_status()
        return r.json()["answer"]

    results = run_evaluation(test_dataset, answer_fn=cloud_answer)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_retriever(collection_name: str | None = None):
    """
    Create a Qdrant retriever.  Uses *collection_name* when provided,
    otherwise falls back to ``settings.QDRANT_COLLECTION_NAME``.
    """
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        collection_name=collection_name or settings.QDRANT_COLLECTION_NAME,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_answer_fn(collection_name: str) -> Callable[[str], str]:
    """
    Build a local RAG chain that answers questions from *collection_name*.

    Use this when you need to evaluate a specific collection (e.g. during
    chunking strategy comparison) without touching the production collection.
    Mirrors the chain in chat.py so results are comparable.
    """
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
    from langchain_core.runnables import RunnablePassthrough  # noqa: PLC0415

    retriever = _build_retriever(collection_name)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

    # Prompt mirrors chat.py exactly so answers are comparable across strategies
    template = """Eres un asistente técnico experto.
    Usa SOLAMENTE el siguiente contexto para responder a la pregunta del usuario.
    Si la respuesta no está en el contexto, di "No tengo información suficiente en los documentos proporcionados".

    Contexto:
    {context}

    Pregunta:
    {question}
    """
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
    return chain.invoke


def _retrieve_contexts(question: str, retriever) -> list[str]:
    """Return the page_content of each chunk retrieved for *question*."""
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    test_dataset: list[dict[str, Any]],
    answer_fn: Callable[[str], str] | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """
    Run RAGAS evaluation on a test dataset.

    Parameters
    ----------
    test_dataset : list[dict]
        Each dict **must** contain:
            question      (str)  the question to ask the RAG system
            ground_truth  (str)  the reference / expected answer
        An optional ``context`` key is accepted but ignored; contexts are
        retrieved live from Qdrant so the evaluation reflects real system
        behaviour.

    answer_fn : callable(question: str) -> str, optional
        Function used to get the RAG system's answer for each question.
        When *None* and *collection_name* is also None, the local
        ``ask_question`` from chat.py is used.
        When *None* but *collection_name* is set, a local chain is built
        against that collection (useful for chunking comparison).
        Pass an HTTP-based callable to target the live Cloud Run endpoint.

    collection_name : str, optional
        Qdrant collection to retrieve contexts from.  Defaults to
        ``settings.QDRANT_COLLECTION_NAME``.  When set and *answer_fn* is
        None, answers are also sourced from this collection.

    Returns
    -------
    dict with three keys:

    per_question : list[dict]
        One entry per sample.  Fields: question, answer, ground_truth,
        contexts (list[str]), scores (dict[metric -> float|None]).

    scores : dict[str, float]
        Mean score per RAGAS metric across all samples.

    overall : float
        Unweighted mean of all metric scores.
    """
    if not test_dataset:
        raise ValueError("test_dataset must not be empty")

    if answer_fn is None:
        if collection_name:
            # Point answers at the same temp collection as context retrieval.
            answer_fn = build_answer_fn(collection_name)
        else:
            # Lazy import keeps module loadable without the full app stack.
            from app.services.chat import ask_question  # noqa: PLC0415
            answer_fn = ask_question

    retriever = _build_retriever(collection_name)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    total = len(test_dataset)
    for i, item in enumerate(test_dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("Sample %d/%d: %s", i + 1, total, question[:70])
        print(f"  [{i + 1}/{total}] {question[:70]}{'...' if len(question) > 70 else ''}")

        answer = answer_fn(question)
        retrieved = _retrieve_contexts(question, retriever)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved)
        ground_truths.append(ground_truth)

    # Build HuggingFace Dataset expected by RAGAS 0.1.x
    hf_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Use the same model family as the RAG system to keep costs predictable
    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )
    eval_embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    raw_result = evaluate(
        hf_dataset,
        metrics=METRICS,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # Mean scores across all samples
    scores: dict[str, float] = {
        name: float(raw_result[name]) for name in METRIC_NAMES
    }
    overall = sum(scores.values()) / len(scores)

    # Per-question breakdown from the result DataFrame
    result_df = raw_result.to_pandas()
    per_question: list[dict] = []
    for idx in range(len(questions)):
        row = result_df.iloc[idx]
        per_question.append(
            {
                "question": questions[idx],
                "answer": answers[idx],
                "ground_truth": ground_truths[idx],
                "contexts": contexts[idx],
                "scores": {
                    name: float(row[name]) if name in row.index else None
                    for name in METRIC_NAMES
                },
            }
        )

    return {
        "per_question": per_question,
        "scores": scores,
        "overall": overall,
    }
