from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
import numpy as np

class MongoDBVectorDB:
    def __init__(self, uri, db_name='rag_chatbot', collection_name='vectors', dimension=384):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        self.dimension = dimension

        # Create vector index if not exists (Atlas Search required for production)
        self.collection.create_index([("embedding", "2dsphere")])

    def add_document(self, doc_id, embedding, metadata=None):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        doc = {
            "_id": str(doc_id),
            "embedding": embedding,
            "metadata": metadata or {}
        }
        self.collection.replace_one({"_id": str(doc_id)}, doc, upsert=True)

    def query(self, query_vector, top_k=5):
        # In Atlas: You’d use $vectorSearch. Locally, we'll simulate with cosine similarity:
        all_docs = list(self.collection.find({}))
        results = []

        def cosine_sim(v1, v2):
            v1, v2 = np.array(v1), np.array(v2)
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            return dot / norm if norm else 0.0

        for doc in all_docs:
            emb = doc.get("embedding", [])
            score = cosine_sim(query_vector, emb)
            results.append({
                "metadata": doc.get("metadata", {}),
                "score": score
            })

        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        return results

    def delete_document(self, doc_id):
        self.collection.delete_one({"_id": str(doc_id)})
