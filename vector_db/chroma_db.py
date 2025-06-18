import os
import chromadb
from chromadb.config import Settings

class ChromaVectorDB:
    def __init__(self, collection_name="default", persist_directory="./chroma_storage"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_document(self, id, embedding, metadata):
        # Add a single document and its embedding
        self.collection.upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[metadata.get("content", "")]
        )
        print(f"✅ ChromaVectorDB: Document {id} added.")

    def query(self, embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=['documents', 'metadatas']
        )

    def list_documents(self):
        return self.collection.get(include=['ids'])

    def clear(self):
        ids = self.collection.get(include=['ids'])['ids']
        self.collection.delete(ids=ids)
        print("✅ ChromaVectorDB: Collection cleared.")
