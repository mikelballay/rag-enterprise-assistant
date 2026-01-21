import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# CAMBIO IMPORTANTE: Usamos la librería especializada, no la de comunidad
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.core.config import settings

def ingest_file(file_path: str):
    """
    Lee un PDF, lo trocea y lo guarda en la base de datos vectorial.
    """
    print(f"📄 Procesando archivo: {file_path}")

    # 1. Cargar el PDF
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {file_path}")
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"   --> Se cargaron {len(docs)} páginas.")

    # 2. Trocear el texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"   --> Se generaron {len(splits)} fragmentos (chunks).")

    # 3. Guardar en Qdrant (Usando la nueva sintaxis)
    print("   --> Generando embeddings y guardando en Qdrant...")
    
    url_qdrant = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    
    # Usamos QdrantVectorStore que es la clase moderna
    QdrantVectorStore.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        url=url_qdrant,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        force_recreate=True 
    )

    print("✅ ¡Ingestión completada con éxito!")