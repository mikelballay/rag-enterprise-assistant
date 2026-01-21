import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env al sistema
load_dotenv()

class Settings:
    PROJECT_NAME: str = "RAG Enterprise Assistant"
    VERSION: str = "1.0.0"
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_portfolio_docs")

    def is_valid(self) -> bool:
        return bool(self.OPENAI_API_KEY)

# Instancia única de configuración para importar en otros archivos
settings = Settings()