import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Claves obligatorias
    OPENAI_API_KEY: str
    
    # Configuración de Qdrant (Ahora flexible para Cloud o Local)
    QDRANT_URL: str  # Ejemplo: "https://xyz...cloud.qdrant.io" o "http://localhost:6333"
    QDRANT_API_KEY: str | None = None  # Puede ser None si estamos en local
    QDRANT_COLLECTION_NAME: str = "rag_portfolio_docs"
    CHUNKING_STRATEGY: str = "RECURSIVE"
    USE_RERANKING: bool = True

    class Config:
        env_file = ".env"

settings = Settings()