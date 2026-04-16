from pydantic import BaseModel

# Lo que el usuario nos envía
class ChatRequest(BaseModel):
    question: str

# Lo que nosotros respondemos
class ChatResponse(BaseModel):
    answer: str
    reranking_enabled: bool

# Para la respuesta de ingestión
class IngestResponse(BaseModel):
    filename: str
    status: str