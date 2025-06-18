#!/usr/bin/env python3
"""
Universal Vector DB Pipeline: multi-vector DB & multi-embedding support

Usage (CLI):
    poetry run python main.py --db faiss --model all-MiniLM-L6-v2 ingest Files/example.pdf
    poetry run python main.py --db chroma --model all-mpnet-base-v2 list
    poetry run python main.py --db pinecone --model roberta-base-nli-mean-tokens watch Files/

Fallback (Interactive):
    python main.py          ← Prompts you to select DB and model, then runs folder watcher
"""

import sys, argparse, os
from document_manager import DocumentManager


def parse_arguments():
    parser = argparse.ArgumentParser(description="Universal Vector DB Pipeline")
    parser.add_argument("--db", choices=["pinecone", "faiss", "chroma"], required=False)
    parser.add_argument("--model", choices=[
        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "distilbert-base-nli-stsb-mean-tokens",
        "bert-base-nli-mean-tokens", "roberta-base-nli-mean-tokens"
    ], required=False)
    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # ingest <file_path>
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("file")

    # query <text>
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query", nargs="+")
    query_parser.add_argument("--top_k", type=int, default=5)

    # list
    subparsers.add_parser("list")

    # delete <file_path>
    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("file")

    # watch <folder_path>
    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("folder")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 🧠 INTERACTIVE fallback
    if not args.command:
        print("🧠 No CLI command given. Launching interactive mode...\n")

        db_map = {"1": "pinecone", "2": "faiss", "3": "chroma"}
        model_map = {
            "1": "all-MiniLM-L6-v2",
            "2": "all-mpnet-base-v2",
            "3": "distilbert-base-nli-stsb-mean-tokens",
            "4": "bert-base-nli-mean-tokens",
            "5": "roberta-base-nli-mean-tokens"
        }

        print("🗃️ Choose your vector database:")
        for k, v in db_map.items():
            print(f" {k}) {v}")
        db_choice = db_map.get(input("Enter choice [1-3]: ").strip(), "pinecone")

        print("\n🤖 Choose an embedding model:")
        for k, v in model_map.items():
            print(f" {k}) {v}")
        model_choice = model_map.get(input("Enter choice [1-5]: ").strip(), "all-MiniLM-L6-v2")

        try:
            doc_manager = DocumentManager(db_type=db_choice, model_name=model_choice)
            folder_path = "Files"
            observer = doc_manager.watch_folder(folder_path)
            print(f"\n✅ Watching folder: {folder_path}")
            print("📂 Drop files here to auto-ingest. Press Ctrl+C to stop.")
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping folder watch.")
            observer.stop()
            observer.join()
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
        return

    # ✅ CLI MODE
    try:
        doc_manager = DocumentManager(db_type=args.db, model_name=args.model)
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "ingest":
        result = doc_manager.ingest_file(args.file)
        if result.get("status") == "ingested":
            print(f"✅ Ingested: {os.path.basename(args.file)} (ID: {result.get('id')})")
        elif result.get("status") == "skipped":
            reason = result.get("reason")
            if reason == "no_change":
                print(f"⚠️ Skipped: {os.path.basename(args.file)} (no changes)")
            elif reason == "duplicate_content":
                print(f"⚠️ Duplicate: Content already indexed")
            elif reason == "empty_file":
                print(f"⚠️ Empty file: Skipped {os.path.basename(args.file)}")
        else:
            print(f"❌ Failed to ingest {os.path.basename(args.file)}", file=sys.stderr)

    elif args.command == "query":
        query_text = " ".join(args.query)
        results = doc_manager.query(query_text, top_k=args.top_k)
        if not results:
            print("❌ No similar documents found.")
        else:
            print(f"🔍 Top {len(results)} results for: \"{query_text}\":")
            for i, res in enumerate(results, 1):
                file_path = res['file_path']
                score = res['score']
                print(f"{i}. {os.path.basename(file_path)}  (score: {score:.3f})")

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
            print(f"⚠️ File not found in index: {os.path.basename(args.file)}")

    elif args.command == "watch":
        try:
            observer = doc_manager.watch_folder(args.folder)
            print(f"👀 Watching: {args.folder}")
            print("📂 Drop files to auto-ingest. Press Ctrl+C to stop.")
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping watcher...")
            observer.stop()
            observer.join()


if __name__ == "__main__":
    main()
