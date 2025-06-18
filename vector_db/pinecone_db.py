from dotenv import load_dotenv
load_dotenv()

import os
print("DEBUG ENV:", os.getenv("PINECONE_API_KEY"), os.getenv("PINECONE_ENV"))

from pinecone import Pinecone, ServerlessSpec


class PineconeVectorDB:
    """Vector database handler for Pinecone v3."""
    def __init__(self, index_name: str, dimension: int):
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")  # should be the region (e.g., "us-west-4")
        if not api_key or not env:
            raise RuntimeError("Pinecone API key and environment must be set (PINECONE_API_KEY, PINECONE_ENV).")

        self.dimension = dimension
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        # Create index if it doesn't exist
        index_names = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in index_names:
            print(f"🆕 Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=env)
            )
        else:
            # Optional: verify dimension
            info = self.pc.describe_index(self.index_name)
            existing_dim = info.dimension if hasattr(info, "dimension") else info.get("dimension")
            if existing_dim != self.dimension:
                raise RuntimeError(f"Index dimension mismatch: expected {self.dimension}, found {existing_dim}")

        self.index = self.pc.Index(self.index_name)

    def add_document(self, doc_id: str, embedding: list, metadata: dict = None):
        """Add or update a document vector in the Pinecone index."""
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        vector = {
            "id": str(doc_id),
            "values": embedding,
            "metadata": metadata or {}
        }
        self.index.upsert(vectors=[vector])
        print(f"📤 Pinecone: Upserted document ID {doc_id}")

    def delete_document(self, doc_id: str):
        """Delete a document by ID from Pinecone."""
        self.index.delete(ids=[str(doc_id)])
        return True

    def query(self, vector: list, top_k: int = 5):
        """Query Pinecone for top-k similar vectors."""
        result = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
        return [{"id": m["id"], "score": m["score"]} if isinstance(m, dict) else {"id": m.id, "score": m.score} for m in matches]

    def list_documents(self):
        """List all IDs in the Pinecone index — not supported, return empty list."""
        return []  # Track this externally if needed
