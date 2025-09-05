#!/usr/bin/env python3
"""
Universal Vector DB Pipeline: multi-vector DB & multi-embedding & multi-LLM support
"""

import sys
import argparse
import os
import time
from dotenv import load_dotenv

load_dotenv()  # ✅ Load environment variables

from document_manager import DocumentManager
from chat.interface import LLMInterface  # ✅ Ensure this exists


def parse_arguments():
    parser = argparse.ArgumentParser(description="Universal Vector DB Pipeline")
    parser.add_argument("--db", choices=["pinecone", "faiss", "chroma", "mongodb", "qdrant"], required=False)
    parser.add_argument("--model", choices=[
        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "distilbert-base-nli-stsb-mean-tokens",
        "bert-base-nli-mean-tokens", "roberta-base-nli-mean-tokens"
    ], required=False)
    parser.add_argument("--llm", choices=["together", "openrouter", "cohere"], required=False)
    parser.add_argument("--llm_model", required=False, help="LLM model name")

    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("file")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query", nargs="+")
    query_parser.add_argument("--top_k", type=int, default=5)

    subparsers.add_parser("list")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("file")

    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("folder")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.command:
        print("🧠 No CLI command given. Launching interactive mode...\n")

        db_map = {"1": "pinecone", "2": "faiss", "3": "chroma", "4": "mongodb", "5": "qdrant"}
        model_map = {
            "1": "all-MiniLM-L6-v2",
            "2": "all-mpnet-base-v2",
            "3": "distilbert-base-nli-stsb-mean-tokens",
            "4": "bert-base-nli-mean-tokens",
            "5": "roberta-base-nli-mean-tokens"
        }
        llm_map = {
            "1": ("together", "mistralai/Mistral-7B-Instruct-v0.2"),
            "2": ("openrouter", "openai/gpt-3.5-turbo"),
            "3": ("cohere", "command-r")
        }

        print("🗃️ Choose your vector database:")
        for k, v in db_map.items():
            print(f" {k}) {v}")
        db_choice = db_map.get(input("Enter choice [1-5]: ").strip(), "pinecone")

        print("\n🤖 Choose an embedding model:")
        for k, v in model_map.items():
            print(f" {k}) {v}")
        model_choice = model_map.get(input("Enter choice [1-5]: ").strip(), "all-MiniLM-L6-v2")

        print("\n📝 Choose an LLM service:")
        for k, (llm_type, llm_model) in llm_map.items():
            print(f" {k}) {llm_type} → {llm_model}")
        llm_choice = input(f"Enter choice [1-{len(llm_map)}]: ").strip()
        llm_type, llm_model = llm_map.get(llm_choice, ("together", "mistralai/Mistral-7B-Instruct-v0.2"))

        observer = None

        try:
            doc_manager = DocumentManager(db_type=db_choice, model_name=model_choice)
            llm_interface = LLMInterface(llm_type=llm_type, model_name=llm_model)

            folder_path = "Files"
            observer = doc_manager.watch_folder(folder_path)
            print(f"\n✅ Watching folder: {folder_path}")
            print("📂 Drop files here to auto-ingest. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping folder watch.")
            if observer:
                observer.stop()
                observer.join()
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
        return

    try:
        doc_manager = DocumentManager(db_type=args.db, model_name=args.model)
        llm_interface = LLMInterface(llm_type=args.llm or "together", model_name=args.llm_model or "mistralai/Mistral-7B-Instruct-v0.2")
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "ingest":
        result = doc_manager.ingest_file(args.file)
        if result.get("status") == "ingested":
            print(f"✅ Ingested: {os.path.basename(args.file)} (ID: {result.get('id')})")
        elif result.get("status") == "skipped":
            print(f"⚠️ Skipped: {os.path.basename(args.file)} ({result.get('reason')})")
        else:
            print(f"❌ Failed to ingest {os.path.basename(args.file)}", file=sys.stderr)

    elif args.command == "query":
        from corrective_rag import grade_passages, reflect_answer

        query_text = " ".join(args.query)
        raw_results = doc_manager.query(query_text, top_k=args.top_k)

        if not raw_results:
            print("❌ No similar documents found.")
            return

        graded = grade_passages(raw_results, query_text)
        top_graded = [doc for doc, score in graded if score > 0.6]

        combined_context = "\n\n".join(doc.get('content', '') for doc in top_graded if 'content' in doc)

        prompt = f"""Using the following source passages, answer the query:
Query: "{query_text}"

Sources:
{combined_context}

Answer:"""

        answer = llm_interface.ask(query_text, context=combined_context)
        final_answer = reflect_answer(answer, combined_context)

        print(f"\n🧠 Final Answer:\n{final_answer}\n")
        print("📚 Sources:")
        for i, doc in enumerate(top_graded, 1):
            print(f"{i}. {os.path.basename(doc.get('file_path', 'unknown'))}  (score: {doc.get('score', 0):.3f})")

        if args.db == "mongodb":
            try:
                doc_manager.vector_db.add_document(
                    doc_id=query_text,
                    embedding=[],
                    metadata={
                        "question": query_text,
                        "context": combined_context,
                        "raw_answer": answer,
                        "final_answer": final_answer
                    }
                )
                print("✅ Interaction saved to MongoDB.")
            except Exception as e:
                print(f"❌ Failed to save to MongoDB: {e}")

    elif args.command == "list":
        docs = doc_manager.list_documents()
        if not docs:
            print("📭 No documents indexed.")
        else:
            print("📄 Indexed files:")
            for path in docs:
                print(f"- {path}")

    elif args.command == "delete":
        result = doc_manager.delete_document(args.file)
        if result.get("status") == "deleted":
            print(f"🗑️ Deleted: {os.path.basename(args.file)}")
        else:
            print(f"⚠️ File not found: {os.path.basename(args.file)}")

    elif args.command == "watch":
        observer = None
        try:
            observer = doc_manager.watch_folder(args.folder)
            print(f"👀 Watching: {args.folder}")
            print("📂 Drop files to auto-ingest. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping watcher...")
            if observer:
                observer.stop()
                observer.join()


if __name__ == "__main__":
    main()
