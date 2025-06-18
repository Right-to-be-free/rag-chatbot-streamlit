import os

class EmbeddingModel:
    """Loads a SentenceTransformer embedding model for generating vector embeddings."""
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install the 'sentence_transformers' package to use embedding models.")
        
        self.model = SentenceTransformer(model_name)

        try:
            self.dim = self.model.get_sentence_embedding_dimension()
        except AttributeError:
            test_vec = self.model.encode("test", convert_to_numpy=True)
            self.dim = len(test_vec)

    def embed_text(self, text: str):
        """Generate an embedding vector for a single piece of text."""
        vec = self.model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_texts(self, texts: list[str]):
        """Generate embedding vectors for a list of text chunks (batch embedding)."""
        vecs = self.model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in vecs]
