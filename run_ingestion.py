from app.services.ingestion import ingest_file

if __name__ == "__main__":
    # Asegúrate de que tienes un archivo llamado 'documento.pdf' en la carpeta data/
    pdf_path = "data/JustificanteSolicitud.pdf"
    
    try:
        ingest_file(pdf_path)
    except Exception as e:
        print(f"❌ Error durante la ingestión: {e}")