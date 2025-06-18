import os, time, json, hashlib, re
import numpy as np
from embedding_model import EmbeddingModel
from vector_db import PineconeVectorDB  

from file_utils import load_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from chunker import chunk_text_semantic


class WatcherHandler(FileSystemEventHandler):
    def __init__(self, doc_manager):
        self.doc_manager = doc_manager
        self.processed_files = set()

    def on_created(self, event):
        if event.is_directory:
            return
        path = os.path.abspath(event.src_path)
        if os.path.basename(path).lower() in ("desktop.ini", ".ds_store"):
            return
        print(f"🌟 File created: {os.path.basename(path)}")

        for i in range(5):
            try:
                with open(path, "rb"):
                    break
            except (PermissionError, OSError):
                print(f"🔒 File still locked ({i+1}/5): {os.path.basename(path)}")
                time.sleep(1)
        else:
            print(f"❌ Skipping file - still locked: {os.path.basename(path)}")
            return

        self.doc_manager.ingest_file(path)
        self.processed_files.add(path)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = os.path.abspath(event.src_path)
        if os.path.basename(path).lower() in ("desktop.ini", ".ds_store"):
            return
        print(f"✏️ File modified: {os.path.basename(path)}")

        for i in range(5):
            try:
                with open(path, "rb"):
                    break
            except (PermissionError, OSError):
                print(f"🔒 File still locked ({i+1}/5): {os.path.basename(path)}")
                time.sleep(1)
        else:
            print(f"❌ Skipping file - still locked: {os.path.basename(path)}")
            return

        self.doc_manager.ingest_file(path)
        self.processed_files.add(path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = os.path.abspath(event.src_path)
        print(f"🗑️ File deleted: {os.path.basename(path)}")
        self.doc_manager.delete_document(path)


class DocumentManager:
    def __init__(self, db_type: str, model_name: str):
        if db_type.lower() != "pinecone":
            raise ValueError("Only 'pinecone' is supported in this deployment.")

        self.db_type = "pinecone"
        self.embedding_model = EmbeddingModel(model_name)
        embed_dim = self.embedding_model.dim

        safe_name = re.sub(r'[^a-z0-9\-]', '-', model_name.lower())
        index_name = f"{safe_name}-{embed_dim}"

        self.vector_db = PineconeVectorDB(index_name=index_name, dimension=embed_dim)

        self.meta_file = f"{index_name}_meta.json"
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                data = json.load(f)
                self.path_to_id = data.get("path_to_id", {})
                self.id_to_path = data.get("id_to_path", {})
                self.path_to_hash = data.get("path_to_hash", {})
                self.hash_to_id = data.get("hash_to_id", {})
        else:
            self.path_to_id, self.id_to_path, self.path_to_hash, self.hash_to_id = {}, {}, {}, {}

    def _save_metadata(self):
        with open(self.meta_file, "w") as f:
            json.dump({
                "path_to_id": self.path_to_id,
                "id_to_path": self.id_to_path,
                "path_to_hash": self.path_to_hash,
                "hash_to_id": self.hash_to_id
            }, f, indent=2)

    def ingest_file(self, file_path: str):
        file_path = os.path.abspath(file_path)
        for i in range(5):
            try:
                content = load_file(file_path)
                break
            except PermissionError:
                print(f"🔁 Retry {i+1}/5: File locked - {os.path.basename(file_path)}")
                time.sleep(1)
        else:
            print(f"❌ Could not load file after retries: {file_path}")
            return {"status": "error", "reason": "permission_denied"}

        if not content.strip():
            return {"status": "skipped", "reason": "empty_file"}

        file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        if file_path in self.path_to_id:
            old_hash = self.path_to_hash.get(file_path)
            if old_hash == file_hash:
                return {"status": "skipped", "reason": "no_change"}
            doc_id = self.path_to_id[file_path]
            self.vector_db.delete_document(doc_id)
            if old_hash and self.hash_to_id.get(old_hash) == doc_id:
                del self.hash_to_id[old_hash]
            new_id = doc_id
        else:
            if file_hash in self.hash_to_id:
                return {"status": "skipped", "reason": "duplicate_content", "duplicate_of": self.hash_to_id[file_hash]}
            new_id = file_path

        chunks = chunk_text_semantic(content, model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = self.embedding_model.embed_texts(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{new_id}_chunk{i}"
            metadata = {
                "file": file_path,
                "chunk_index": i,
                "chunk_text": chunk[:500],
                "hash": file_hash
            }

            self.vector_db.add_document(chunk_id, vec, metadata=metadata)

        self.path_to_id[file_path] = new_id
        self.id_to_path[str(new_id)] = file_path
        self.path_to_hash[file_path] = file_hash
        self.hash_to_id[file_hash] = new_id
        self._save_metadata()

        return {"status": "ingested", "id": new_id, "chunks": len(chunks)}

    def delete_document(self, file_path: str):
        file_path = os.path.abspath(file_path)
        if file_path not in self.path_to_id:
            return {"status": "error", "reason": "not_found"}

        doc_id = self.path_to_id[file_path]
        self.vector_db.delete_document(doc_id)

        file_hash = self.path_to_hash.get(file_path)
        if file_hash and self.hash_to_id.get(file_hash) == doc_id:
            del self.hash_to_id[file_hash]

        del self.path_to_id[file_path]
        del self.path_to_hash[file_path]
        if str(doc_id) in self.id_to_path:
            del self.id_to_path[str(doc_id)]

        self._save_metadata()
        return {"status": "deleted", "id": doc_id}

    def list_documents(self):
        return sorted(self.path_to_id.keys())

    def query(self, query_text: str, top_k: int = 5):
        query_embedding = self.embedding_model.embed_text(query_text)
        return self.vector_db.query(query_embedding, top_k=top_k)

    def retrieve(self, query_text: str, top_k: int = 5):
        return self.query(query_text, top_k=top_k)

    def watch_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("🔄 Scanning existing files...")
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if not os.path.isfile(full_path): continue
            if filename.lower() in ("desktop.ini", ".ds_store"): continue

            result = self.ingest_file(full_path)
            if result.get("status") == "ingested":
                print(f"✅ Ingested existing: {filename}")
            elif result.get("status") == "skipped":
                reason = result.get("reason")
                if reason == "no_change":
                    print(f"↪️ Skipped (no change): {filename}")
                elif reason == "duplicate_content":
                    print(f"⚠️ Duplicate: {filename}")
                elif reason == "empty_file":
                    print(f"⚠️ Empty file: {filename}")

        observer = Observer()
        observer.schedule(WatcherHandler(self), folder_path, recursive=False)
        observer.start()
        return observer
