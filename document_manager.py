import os, time, json, hashlib, re, uuid
import numpy as np
import pandas as pd
from embedding_model import EmbeddingModel
from vector_db import PineconeVectorDB, FaissVectorDB, ChromaVectorDB, MongoDBVectorDB, QdrantVectorDB
from file_utils import load_file
from chunker import chunk_text_semantic

class DocumentManager:
    def __init__(self, db_type: str, model_name: str):
        self.db_type = db_type.lower()
        self.embedding_model = EmbeddingModel(model_name)
        self.embed_dim = self.embedding_model.dim

        safe_name = re.sub(r'[^a-z0-9\-]', '-', model_name.lower())
        self.index_name = (
            f"{safe_name}-{self.embed_dim}" if self.db_type == "pinecone"
            else f"{self.db_type}_{safe_name}_{self.embed_dim}"
        )

        self.vector_db = self._init_vector_db()
        print(f"✅ Vector DB selected: {type(self.vector_db).__name__}")

        self.meta_file = f"{self.index_name}_meta.json"
        self._normalize = (self.db_type == "faiss")
        self._load_metadata()
        self._id_counter = max(self.path_to_id.values(), default=0) if self.db_type == "faiss" else 0

    def _init_vector_db(self):
        if self.db_type == "pinecone":
            return PineconeVectorDB(index_name=self.index_name, dimension=self.embed_dim)
        elif self.db_type == "faiss":
            return FaissVectorDB(dimension=self.embed_dim, model_name=self.embedding_model.model_name)
        elif self.db_type == "chroma":
            return ChromaVectorDB(collection_name=self.index_name)
        elif self.db_type == "mongodb":
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise ValueError("MONGO_URI is not set in environment variables.")
            print(f"✅ MongoDB URI loaded: {mongo_uri}")
            return MongoDBVectorDB(uri=mongo_uri, db_name='rag_chatbot', collection_name=self.index_name, dimension=self.embed_dim)
        elif self.db_type == "qdrant":
            return QdrantVectorDB(collection_name=self.index_name, dimension=self.embed_dim)
        else:
            raise ValueError(f"Unsupported vector DB type: {self.db_type}")

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
        file_path = os.path.abspath(file_path).lower()

        try:
            if file_path.endswith(('.csv', '.xlsx')):
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                df = df[df.columns.sort_values()]
                df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
                content = df.to_csv(index=False)
            else:
                content = load_file(file_path)
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {str(e)}")
            return {"status": "error", "reason": str(e)}

        if not content.strip():
            return {"status": "skipped", "reason": "empty_file"}

        file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        new_id, is_update = self._handle_deduplication(file_path, file_hash)

        if new_id is None:
            print(f"↪️ Skipped: {os.path.basename(file_path)} (duplicate_content)")
            return {"status": "skipped", "reason": "duplicate_content"}

        chunks = chunk_text_semantic(content, model_name=self.embedding_model.model_name)
        embeddings = self.embedding_model.embed_texts(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            # ✅ For Qdrant use UUIDs; for others keep previous logic
            if self.db_type == "qdrant":
                chunk_id = str(uuid.uuid4())
            elif self.db_type == "faiss":
                chunk_id = self._id_counter + i
            else:
                chunk_id = f"{new_id}_chunk{i}"

            if self._normalize:
                vec = np.array(vec, dtype='float32')
                norm = np.linalg.norm(vec)
                vec = (vec / norm).tolist() if norm != 0 else vec.tolist()

            self.vector_db.add_document(chunk_id, vec, metadata={
                "chunk_index": i,
                "chunk_text": chunk[:500],
                "hash": file_hash
            })

        if self.db_type == "faiss":
            self._id_counter += len(chunks)

        self._update_metadata(file_path, new_id, file_hash)
        print(f"✅ Ingested: {os.path.basename(file_path)} ({len(chunks)} chunks)")
        return {"status": "ingested", "id": new_id, "chunks": len(chunks)}

    def _handle_deduplication(self, file_path, file_hash):
        if file_path in self.path_to_id:
            old_hash = self.path_to_hash.get(file_path)
            if old_hash == file_hash:
                return None, False
            doc_id = self.path_to_id[file_path]
            self.delete_document(file_path)  # Reuse delete method
            return doc_id, True
        elif file_hash in self.hash_to_id:
            return None, False
        else:
            doc_id = self._id_counter + 1 if self.db_type == "faiss" else file_path
            return doc_id, False

    def delete_document(self, file_path: str):
        file_path = os.path.abspath(file_path).lower()
        if file_path not in self.path_to_id:
            return {"status": "error", "reason": "not_found"}

        doc_id = self.path_to_id[file_path]
        file_hash = self.path_to_hash.get(file_path)

        # ✅ Special case: Qdrant deletes all points by file hash
        if self.db_type == "qdrant":
            try:
                self.vector_db.delete_document_by_metadata("hash", file_hash)
            except Exception as e:
                print(f"❌ Qdrant deletion failed: {e}")
                return {"status": "error", "reason": "qdrant_delete_failed"}
        else:
            self.vector_db.delete_document(doc_id)

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
        if self._normalize:
            vec = np.array(query_embedding, dtype='float32')
            norm = np.linalg.norm(vec)
            query_embedding = (vec / norm).tolist() if norm != 0 else vec.tolist()

        results = self.vector_db.query(query_embedding, top_k=top_k)
        clean_results = []
        for i, res in enumerate(results):
            chunk_text = res.get('metadata', {}).get('chunk_text', '')
            chunk_preview = chunk_text.strip().replace('\n', ' ')[:300]
            print(f"\n📄 [Result {i+1}] {chunk_preview}")
            clean_results.append(res)
        return results
    def _update_metadata(self, path, doc_id, file_hash):
        self.path_to_id[path] = doc_id
        self.id_to_path[str(doc_id)] = path
        self.path_to_hash[path] = file_hash
        self.hash_to_id[file_hash] = doc_id
        self._save_metadata()


    def watch_folder(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("🔄 Scanning existing files...")
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if not os.path.isfile(full_path) or filename.lower() in ("desktop.ini", ".ds_store"):
                continue
            result = self.ingest_file(full_path)
            if result.get("status") == "ingested":
                print(f"✅ Ingested existing: {filename}")
            elif result.get("status") == "skipped":
                print(f"↪️ Skipped: {filename} ({result.get('reason')})")

        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class WatcherHandler(FileSystemEventHandler):
            def __init__(self, doc_manager):
                self.doc_manager = doc_manager

            def _handle_file_event(self, path, label):
                if os.path.isdir(path) or os.path.basename(path).lower() in ("desktop.ini", ".ds_store"):
                    return
                print(f"{label} File: {os.path.basename(path)}")
                for i in range(5):
                    try:
                        with open(path, "rb"):
                            break
                    except (PermissionError, OSError):
                        time.sleep(1)
                else:
                    print(f"❌ Skipping file - still locked: {os.path.basename(path)}")
                    return
                self.doc_manager.ingest_file(path)

            def on_created(self, event):
                self._handle_file_event(os.path.abspath(event.src_path).lower(), "🌟 Created")

            def on_modified(self, event):
                self._handle_file_event(os.path.abspath(event.src_path).lower(), "✏️ Modified")

            def on_deleted(self, event):
                path = os.path.abspath(event.src_path).lower()
                if not os.path.isdir(path):
                    print(f"🗑️ Deleted File: {os.path.basename(path)}")
                    self.doc_manager.delete_document(path)

        observer = Observer()
        observer.schedule(WatcherHandler(self), folder_path, recursive=False)
        observer.start()
        return observer
