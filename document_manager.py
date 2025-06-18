import os, time, json, hashlib, re
import numpy as np
from embedding_model import EmbeddingModel
from vector_db.pinecone_db import PineconeVectorDB  # ✅ Direct import

from file_utils import load_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from chunker import chunk_text_semantic


class WatcherHandler(FileSystemEventHandler):
    def __init__(self, doc_manager):
        self.doc_manager = doc_manager

    def on_created(self, event):
        if not event.is_directory:
            self.doc_manager.ingest_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.doc_manager.ingest_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.doc_manager.delete_document(event.src_path)


class DocumentManager:
    def __init__(self, db_type: str, model_name: str):
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
                self.path_to_hash = data.get("path_to_hash", {})
                self.hash_to_id = data.get("hash_to_id", {})
        else:
            self.path_to_hash, self.hash_to_id = {}, {}

    def _save_metadata(self):
        with open(self.meta_file, "w") as f:
            json.dump({
                "path_to_hash": self.path_to_hash,
                "hash_to_id": self.hash_to_id
            }, f, indent=2)

    def ingest_file(self, file_path: str):
        file_path = os.path.abspath(file_path)
        try:
            content = load_file(file_path)
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return

        if not content.strip():
            print(f"⚠️ Empty file skipped: {file_path}")
            return

        file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        if self.path_to_hash.get(file_path) == file_hash:
            print(f"↪️ Skipped (no change): {file_path}")
            return

        doc_id = file_path
        chunks = chunk_text_semantic(content, model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = self.embedding_model.embed_texts(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk{i}"
            metadata = {
                "file": file_path,
                "chunk_index": i,
                "chunk_text": chunk[:500],
                "hash": file_hash
            }
            self.vector_db.add_document(chunk_id, vec, metadata)

        self.path_to_hash[file_path] = file_hash
        self.hash_to_id[file_hash] = doc_id
        self._save_metadata()

        print(f"✅ Ingested: {file_path}")

    def delete_document(self, file_path: str):
        file_path = os.path.abspath(file_path)
        doc_id = file_path
        self.vector_db.delete_document(doc_id)
        self.path_to_hash.pop(file_path, None)
        print(f"🗑️ Deleted: {file_path}")
        self._save_metadata()

    def query(self, query_text: str, top_k: int = 5):
        embedding = self.embedding_model.embed_text(query_text)
        return self.vector_db.query(embedding, top_k=top_k)

    def watch_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                self.ingest_file(full_path)

        observer = Observer()
        observer.schedule(WatcherHandler(self), folder_path, recursive=False)
        observer.start()
        return observer
