from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from app.core.config import settings

def get_rag_chain():
    """
    Configura y devuelve la cadena de procesamiento RAG.
    """
    
    # 1. Configurar el Modelo de Lenguaje (El cerebro)
    llm = ChatOpenAI(
        model="gpt-4o-mini", # O gpt-3.5-turbo si prefieres
        temperature=0,       # 0 = Máxima precisión, 0 creatividad (Ideal para técnico)
        api_key=settings.OPENAI_API_KEY
    )

    # 2. Conectar a la Base de Datos (La memoria)
    url_qdrant = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        url=url_qdrant
    )
    
    # Convertimos la DB en un "Retriever" (buscador)
    # k=3 significa: "Traeme los 3 fragmentos más relevantes"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. El Prompt (Las instrucciones)
    template = """Eres un asistente técnico experto. 
    Usa SOLAMENTE el siguiente contexto para responder a la pregunta del usuario.
    Si la respuesta no está en el contexto, di "No tengo información suficiente en los documentos proporcionados".
    
    Contexto:
    {context}
    
    Pregunta:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. La Cadena (El pipeline)
    # Explicación:
    # 1. Toma la pregunta y busca contexto (retriever)
    # 2. Pasa el contexto y la pregunta al prompt
    # 3. Pasa el prompt al LLM
    # 4. Parsea la salida a texto simple
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(question: str):
    chain = get_rag_chain()
    response = chain.invoke(question)
    return response