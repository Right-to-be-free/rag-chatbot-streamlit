from document_manager import DocumentManager
from chat.interface import LLMInterface

def chat_loop():
    db_type = input("🗃️ Choose vector DB (pinecone/faiss/chroma): ").strip().lower()
    model_name = input("🤖 Enter embedding model (e.g., all-MiniLM-L6-v2): ").strip()

    doc_manager = DocumentManager(db_type=db_type, model_name=model_name)
    llm = LLMInterface()

    print("\n💬 Ask questions (type 'exit' to quit):")
    while True:
        query = input("\n🧠 You: ")
        if query.lower() in ['exit', 'quit']:
            print("👋 Exiting chat.")
            break

        results = doc_manager.query(query, top_k=3)
        context = "\n".join([f"- {r['file_path']}" for r in results])

        answer = llm.ask(query, context=context)
        print(f"\n🤖 LLM: {answer}")
