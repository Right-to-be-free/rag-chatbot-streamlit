import os
import pickle
import numpy as np
import faiss


class FaissVectorDB:
    def __init__(self, dimension, model_name):
        self.dimension = dimension
        safe_model = model_name.replace("/", "_").replace("-", "_")
        self.index_path = f"faiss_{safe_model}_{dimension}.index"
        self.ids_path = f"{self.index_path}.ids"

        if os.path.exists(self.index_path):
            print(f"📦 Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            if self.index.d != self.dimension:
                raise ValueError(f"❌ FAISS index dimension mismatch: index has {self.index.d}, expected {self.dimension}")
        else:
            print(f"🆕 Creating new FAISS index with dimension {self.dimension}")
            self.index = faiss.IndexFlatL2(self.dimension)

        # Load or initialize document ID tracking
        if os.path.exists(self.ids_path):
            with open(self.ids_path, "rb") as f:
                self.ids = pickle.load(f)
        else:
            self.ids = []

        self.id_map = {}  # optional metadata store (you can persist this too if needed)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "wb") as f:
            pickle.dump(self.ids, f)

    def add_document(self, doc_id, embedding, metadata=None):
        vector = [embedding]

        if len(vector[0]) != self.index.d:
            raise ValueError(f"❌ Embedding dimension mismatch: got {len(vector[0])}, expected {self.index.d}")

        self.index.add(np.array(vector).astype('float32'))
        self.ids.append(doc_id)
        self.id_map[doc_id] = metadata
        self.save()
        print(f"✅ FAISS: Document {doc_id} added.")

    def query(self, query_embedding, top_k=5):
        vector = [query_embedding]

        if len(vector[0]) != self.index.d:
            raise ValueError(f"❌ Query dimension mismatch: got {len(vector[0])}, expected {self.index.d}")

        D, I = self.index.search(np.array(vector).astype('float32'), top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            if 0 <= i < len(self.ids):
                doc_id = self.ids[i]
                results.append({"id": doc_id, "score": float(score)})
        return results

    def delete_document(self, doc_id):
        if doc_id not in self.ids:
            print(f"⚠️ FAISS: Document ID {doc_id} not found.")
            return

        index = self.ids.index(doc_id)
        print(f"🗑️ FAISS: Removing document ID {doc_id} at index {index}")
        del self.ids[index]
        del self.id_map[doc_id]

        # Rebuild index from remaining vectors
        new_index = faiss.IndexFlatL2(self.dimension)
        new_vectors = []

        for i in self.ids:
            meta = self.id_map.get(i)
            if meta and "embedding" in meta:
                new_vectors.append(meta["embedding"])

        if new_vectors:
            new_index.add(np.array(new_vectors).astype('float32'))

        self.index = new_index
        self.save()
