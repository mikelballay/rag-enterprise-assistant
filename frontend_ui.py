import streamlit as st
import requests
import os

# CONFIGURACIÓN
# Intenta leer la variable de entorno API_URL. 
# Si no existe (estás en local), usa la del backend de GCP.
API_URL = os.getenv("API_URL", "https://api-rag-82274106778.us-central1.run.app") 

st.set_page_config(page_title="RAG Enterprise Assistant", page_icon="🧠")


def main():
    st.title("🤖 RAG Enterprise Assistant")
    st.markdown("""
    Este asistente utiliza **RAG (Retrieval Augmented Generation)** para responder preguntas
    basadas estrictamente en tus documentos PDF.
    
    *Tecnologías: FastAPI, Qdrant, LangChain, GPT-4o.*
    """)

    # --- BARRA LATERAL (Subida de Archivos) ---
    with st.sidebar:
        st.header("📂 Ingesta de Conocimiento")
        uploaded_file = st.file_uploader("Sube un PDF", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("Procesar e Ingestar"):
                # Warm-up: wake the Cloud Run container before uploading
                with st.spinner("Conectando con el servidor (puede tardar ~30s la primera vez)..."):
                    try:
                        requests.get(f"{API_URL}/", timeout=90)
                    except Exception:
                        pass  # proceed anyway; the ingest call will surface the real error

                with st.spinner("Troceando, vectorizando e indexando..."):
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    try:
                        response = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
                        if response.status_code == 200:
                            st.success("✅ ¡Documento aprendido exitosamente!")
                        else:
                            st.error(f"❌ Error: {response.text}")
                    except requests.exceptions.Timeout:
                        st.error("❌ El servidor tardó demasiado. Inténtalo de nuevo.")
                    except Exception as e:
                        st.error(f"Error de conexión: {e}")

    # --- CHAT PRINCIPAL ---
    
    # 1. Mantener historial en la sesión de Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre el documento..."):
        # Guardar y mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4. Llamar a TU API para obtener respuesta
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Pensando... 🧠")
            
            try:
                # Llamada al endpoint /chat
                payload = {"question": prompt}
                response = requests.post(f"{API_URL}/chat", json=payload, timeout=120)

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    message_placeholder.error(f"Error del servidor: {response.text}")
            except requests.exceptions.Timeout:
                message_placeholder.error("❌ El servidor tardó demasiado. Inténtalo de nuevo.")
            except Exception as e:
                message_placeholder.error(f"No se pudo conectar con la API: {e}")

if __name__ == "__main__":
    main()
