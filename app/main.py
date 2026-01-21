import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import ChatRequest, ChatResponse, IngestResponse
from app.services.chat import ask_question
from app.services.ingestion import ingest_file

app = FastAPI(
    title="RAG Enterprise API",
    version="1.0.0",
    description="API profesional para chatear con documentos PDF"
)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "RAG API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Endpoint para hacer preguntas al documento.
    """
    try:
        response_text = ask_question(request.question)
        return ChatResponse(answer=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(file: UploadFile = File(...)):
    """
    Endpoint para subir un PDF y procesarlo automáticamente.
    """
    # Guardamos el archivo temporalmente
    temp_file_path = f"data/{file.filename}"
    
    # Nos aseguramos de que la carpeta data existe
    os.makedirs("data", exist_ok=True)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Llamamos a tu lógica de ingestión
        ingest_file(temp_file_path)
        
        # Limpieza opcional: Borrar el archivo después de procesar
        # os.remove(temp_file_path) 
        
        return IngestResponse(filename=file.filename, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")