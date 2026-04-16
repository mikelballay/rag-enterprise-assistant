from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.core.config import settings

# Retrieval config
_RETRIEVAL_K_WITH_RERANKING = 10   # wide net before reranking
_RETRIEVAL_K_WITHOUT_RERANKING = 3  # original behaviour
_RERANKER_TOP_N = 3

_PROMPT_TEMPLATE = """Eres un asistente técnico experto.
    Usa SOLAMENTE el siguiente contexto para responder a la pregunta del usuario.
    Si la respuesta no está en el contexto, di "No tengo información suficiente en los documentos proporcionados".

    Contexto:
    {context}

    Pregunta:
    {question}
    """


def _build_vectorstore():
    return QdrantVectorStore.from_existing_collection(
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def ask_question_full(question: str) -> dict:
    """
    Run the full RAG pipeline and return both the answer and metadata.

    Returns
    -------
    dict with keys:
        answer            (str)   the LLM-generated answer
        reranking_enabled (bool)  whether the reranking step was applied
    """
    use_reranking: bool = settings.USE_RERANKING

    # ── 1. Retrieve ────────────────────────────────────────────────────────
    k = _RETRIEVAL_K_WITH_RERANKING if use_reranking else _RETRIEVAL_K_WITHOUT_RERANKING
    retriever = _build_vectorstore().as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    # ── 2. Rerank (optional) ───────────────────────────────────────────────
    if use_reranking:
        from app.services.reranker import get_reranker  # lazy import keeps startup fast
        texts = get_reranker().rerank(
            query=question,
            documents=[d.page_content for d in docs],
            top_n=_RERANKER_TOP_N,
        )
    else:
        texts = [d.page_content for d in docs]

    context = "\n\n".join(texts)

    # ── 3. Generate ────────────────────────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "reranking_enabled": use_reranking}


def ask_question(question: str) -> str:
    """
    Convenience wrapper — returns only the answer string.

    Keeps the same signature as before so evaluation pipelines and CLI
    scripts that call ``ask_question`` continue to work unchanged.
    """
    return ask_question_full(question)["answer"]


# ---------------------------------------------------------------------------
# Legacy helper kept for backward compatibility
# ---------------------------------------------------------------------------

def get_rag_chain():
    """
    Returns the original LangChain chain (retriever k=3, no reranking).

    Retained so any external code that imported this function continues to
    work.  New code should call ``ask_question`` or ``ask_question_full``.
    """
    from langchain_core.runnables import RunnablePassthrough  # noqa: PLC0415

    retriever = _build_vectorstore().as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
