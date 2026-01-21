from app.services.chat import ask_question

if __name__ == "__main__":
    print("🤖 BIENVENIDO AL CHAT RAG (Escribe 'salir' para terminar)")
    print("---------------------------------------------------------")
    
    while True:
        user_query = input("\nPregunta algo sobre tu documento: ")
        
        if user_query.lower() in ['salir', 'exit', 'quit']:
            break
            
        print("Pensando... 🤔")
        try:
            response = ask_question(user_query)
            print(f"\nRespuesta:\n{response}")
        except Exception as e:
            print(f"❌ Error: {e}")