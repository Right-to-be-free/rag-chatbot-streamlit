import os
import time
from chat.interface import LLMInterface
from document_manager import DocumentManager

def chat_loop():
    # Vector DB Options
    db_options = {
        "1": "pinecone",
        "2": "faiss",
        "3": "chroma",
        "4": "mongodb",
        "5": "qdrant"
    }

    # Embedding Model Options
    embedding_options = {
        "1": "all-MiniLM-L6-v2",
        "2": "all-mpnet-base-v2",
        "3": "distilbert-base-nli-stsb-mean-tokens",
        "4": "bert-base-nli-mean-tokens",
        "5": "roberta-base-nli-mean-tokens"
    }

    # LLM Options
    llm_options = {
        "1": ("together", "mistralai/Mistral-7B-Instruct-v0.2"),
        "2": ("openrouter", "openai/gpt-3.5-turbo"),
        "3": ("cohere", "command-r-plus"),
    }

    print("\n🗃️ Choose Vector Database:")
    for key, val in db_options.items():
        print(f" {key}) {val}")
    db_choice = db_options.get(input("Enter choice [1-5]: ").strip(), "pinecone")

    print("\n🤖 Choose Embedding Model:")
    for key, val in embedding_options.items():
        print(f" {key}) {val}")
    embed_choice = embedding_options.get(input("Enter choice [1-5]: ").strip(), "all-MiniLM-L6-v2")

    print("\n📝 Choose LLM:")
    for key, (llm_type, llm_model) in llm_options.items():
        print(f" {key}) {llm_type} → {llm_model}")
    llm_choice = llm_options.get(input("Enter choice [1-3]: ").strip(), ("together", "mistralai/Mistral-7B-Instruct-v0.2"))

    # Initialize Objects
    doc_manager = DocumentManager(db_type=db_choice, model_name=embed_choice)
    llm = LLMInterface(llm_type=llm_choice[0], model_name=llm_choice[1])

    print("\n✅ Ready! Type your question below (or type 'exit' to quit).")

    while True:
        query = input("\n🧠 You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # Query Documents
        results = doc_manager.query(query, top_k=5)
        if not results:
            print("⚠️ No relevant documents found.")
            continue

        # Build Context
        context = "\n\n".join([
            doc.get('metadata', {}).get('chunk_text', '') for doc in results
        ])

        print("\n📄 Top Context Snippets:")
        for i, doc in enumerate(results):
            snippet = doc.get('metadata', {}).get('chunk_text', '')[:300]
            print(f" [{i+1}] {snippet}\n")

        # Generate Answer
        print("💭 Generating answer...")
        response = llm.ask(query, context=context)
        print(f"\n💬 Answer: {response}\n")
