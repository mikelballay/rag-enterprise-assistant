# 🧠 RAG Enterprise Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![LangChain](https://img.shields.io/badge/LangChain-v0.1-orange) ![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red) ![Docker](https://img.shields.io/badge/Container-Docker-blue)

An End-to-End **Retrieval-Augmented Generation (RAG)** system designed for enterprise use cases. It allows users to ingest internal documentation (PDFs) and ask questions with high precision, strictly avoiding hallucinations by grounding answers in the provided context.

## 🚀 Key Features

* **Full-Stack AI Architecture:** From data ingestion to frontend visualization.
* **Vector Database Integration:** Uses **Qdrant** (running via Docker) for efficient semantic search.
* **Anti-Hallucination Guardrails:** The system is prompted to strictly answer *only* using retrieved context.
* **Automated Pipeline:** PDF parsing, chunking, and embedding generation (OpenAI Embeddings).
* **API-First Design:** Robust REST API built with **FastAPI**.
* **Interactive UI:** User-friendly Chat Interface built with **Streamlit**.

## 🛠️ Tech Stack

* **Core:** Python 3.12
* **API:** FastAPI & Uvicorn
* **Orchestration:** LangChain (LCEL)
* **Vector Store:** Qdrant (Dockerized)
* **LLM:** OpenAI GPT-4o / GPT-3.5-turbo
* **Frontend:** Streamlit
* **Environment:** Docker Compose

## 🏗️ Architecture Overview

1. **Ingestion:** PDFs are uploaded via API/UI -> Text Chunking -> Vector Embeddings.
2. **Storage:** Vectors are stored in a local Qdrant container.
3. **Retrieval:** User queries are converted to vectors -> Semantic Search in Qdrant.
4. **Generation:** Relevant context + Query are sent to GPT-4 -> Answer generation.

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-enterprise-assistant.git
cd rag-enterprise-assistant
```

### 2. Environment Configuration
Create a `.env` file in the root directory:

```ini
OPENAI_API_KEY=sk-proj-your-key-here...
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_portfolio_docs
```

### 3. Start the Vector Database
Use Docker Compose to spin up the Qdrant service:

```bash
docker-compose up -d
```

### 4. Install Dependencies
It is recommended to use a virtual environment:

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application
You need to run both the Backend and Frontend (in separate terminals).

### Terminal 1: Backend API

```bash
uvicorn app.main:app --reload
```
API Docs available at: http://127.0.0.1:8000/docs

### Terminal 2: Frontend UI

```bash
streamlit run frontend_ui.py
```
Access the UI at: http://localhost:8501

## 🧪 Usage Example
1. Open the Streamlit UI.
2. Upload a PDF document (e.g., an invoice, a manual, or a report).
3. Click "Procesar e Ingestar".
4. Ask specific questions about the content (e.g., "What is the total amount in the invoice?").
5. Observe how the system retrieves the exact answer from the text.

Developed by Mikel Ballay 
