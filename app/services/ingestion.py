import os
from enum import Enum

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


class ChunkingStrategy(str, Enum):
    FIXED = "FIXED"          # Current behaviour: RecursiveCharacterTextSplitter 1000/200
    RECURSIVE = "RECURSIVE"  # Improved recursive splits: smaller chunks, richer separators
    SEMANTIC = "SEMANTIC"    # Embedding-based semantic chunking (langchain-experimental)


def _build_splitter(strategy: ChunkingStrategy, embeddings):
    """
    Factory that returns the appropriate text splitter for *strategy*.
    *embeddings* is only used by SEMANTIC; pass None for the other two.
    """
    if strategy == ChunkingStrategy.FIXED:
        # Preserves the original ingestion behaviour exactly.
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    if strategy == ChunkingStrategy.RECURSIVE:
        # Smaller chunks + sentence-aware separators for higher precision retrieval.
        return RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        )

    if strategy == ChunkingStrategy.SEMANTIC:
        try:
            from langchain_experimental.text_splitter import SemanticChunker  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Install langchain-experimental to use SEMANTIC chunking: "
                "pip install langchain-experimental"
            ) from exc
        return SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
        )

    raise ValueError(f"Unknown ChunkingStrategy: {strategy}")


def ingest_file(
    file_path: str,
    strategy: ChunkingStrategy | None = None,
    collection_name: str | None = None,
) -> dict:
    """
    Load a PDF, split it and store vectors in Qdrant.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.
    strategy : ChunkingStrategy, optional
        Splitting strategy to use.  Defaults to ``settings.CHUNKING_STRATEGY``.
    collection_name : str, optional
        Qdrant collection to write into.  Defaults to
        ``settings.QDRANT_COLLECTION_NAME``.  Useful for strategy comparison
        where each strategy writes to its own temporary collection.

    Returns
    -------
    dict
        ``{"chunks": int, "avg_chunk_size": int}`` — useful for comparison
        reports.  Existing callers that ignore the return value are unaffected.
    """
    if strategy is None:
        strategy = ChunkingStrategy(settings.CHUNKING_STRATEGY.upper())
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION_NAME

    print(f"Procesando archivo: {file_path}  [strategy={strategy.value}]")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {file_path}")

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"   --> {len(docs)} páginas cargadas.")

    # 2. Build embeddings (needed by SEMANTIC; created once and reused below)
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    # 3. Split
    splitter = _build_splitter(strategy, embeddings)
    splits = splitter.split_documents(docs)
    avg_chunk_size = (
        int(sum(len(d.page_content) for d in splits) / len(splits)) if splits else 0
    )
    print(f"   --> {len(splits)} fragmentos generados  (avg {avg_chunk_size} chars).")

    # 4. Embed + store
    print("   --> Guardando en Qdrant...")
    QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=True,
    )

    print("Ingestión completada.")
    return {"chunks": len(splits), "avg_chunk_size": avg_chunk_size}
