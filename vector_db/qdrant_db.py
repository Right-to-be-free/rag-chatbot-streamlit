from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
import uuid
import numpy as np
import os

class QdrantVectorDB:
    def __init__(self, collection_name, dimension=384):
        self.collection_name = collection_name
        self.dimension = dimension

        # ✅ Read Qdrant API key and URL from environment variables
        qdrant_url = os.getenv("QDRANT_URL")   # e.g., 'https://abc-123-456-789-xyz.us-east-1.aws.cloud.qdrant.io'
        qdrant_api_key = os.getenv("QDRANT_API_KEY")  # your Qdrant API Key

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("❌ QDRANT_URL and QDRANT_API_KEY must be set in the environment variables.")

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        collections = self.client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )

    def add_document(self, doc_id, embedding, metadata=None):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # Qdrant Cloud requires valid IDs → generate UUID if needed
        try:
            point_id = str(uuid.UUID(doc_id))
        except:
            point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata or {}
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def query(self, query_vector, top_k=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        formatted = []
        for hit in results:
            meta = hit.payload or {}
            meta['score'] = hit.score
            formatted.append({'metadata': meta})
        return formatted

    def delete_document(self, doc_id):
        # Caution: this uses exact ID—ensure you store & pass correct UUIDs
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"points": [doc_id]}
        )

    def delete_document_by_metadata(self, field_key, field_value):
        condition = FieldCondition(
            key=field_key,
            match=MatchValue(value=field_value)
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[condition])
        )
