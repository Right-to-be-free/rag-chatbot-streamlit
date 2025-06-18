import argparse
from document_manager import DocumentManager
from rag_utils import build_prompt
from llm_api import generate_from_api  # Or `load_mistral_model()` if using local

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--db", required=True, choices=["faiss", "chroma", "pinecone"], help="Vector DB to query.")
    parser.add_argument("--model", required=True, help="Embedding model used for retrieval.")
    parser.add_argument("--question", required=True, help="User query/question.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve.")

    args = parser.parse_args()

    print(f"📥 Querying top {args.top_k} chunks for:\n❓ {args.question}\n")

    # Load vector DB and embed the query
    doc_manager = DocumentManager(db_type=args.db, model_name=args.model)
    results = doc_manager.query(args.question, top_k=args.top_k)

    if not results:
        print("⚠️ No matching documents found.")
        return

    # Extract context text
    chunks = []
    for result in results:
        file_path = result["file_path"]
        score = result["score"]
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read())

    # Format prompt
    prompt = build_prompt(args.question, chunks)

    # Generate answer from LLM
    answer = generate_from_api(prompt)  # Or call local model
    print(f"\n🤖 Answer:\n{answer}")

if __name__ == "__main__":
    main()
