import re
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

nltk.download("punkt")

# Token limit mapping for embedding models
MODEL_TOKEN_LIMITS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 384,
    "distilbert-base-nli-stsb-mean-tokens": 512,
    "bert-base-nli-mean-tokens": 512,
    "roberta-base-nli-mean-tokens": 512
}

def chunk_text_semantic(text: str, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, overlap: int = 45, verbose: bool = True):
    """
    Splits text into semantically meaningful chunks using LangChain's recursive splitter + token-aware logic.

    Args:
        text (str): Full input text.
        model_name (str): Embedding model name (can be full Hugging Face path).
        chunk_size (int): Character-level chunk size.
        overlap (int): Character-level overlap between chunks.
        verbose (bool): Whether to show warnings for sentence overflows.

    Returns:
        List[str]: List of text chunks ready for embedding.
    """
    model_key = model_name.split("/")[-1]
    token_limit = MODEL_TOKEN_LIMITS.get(model_key, 512)

    # Load tokenizer to simulate token count if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Clean and normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Use LangChain's semantic-aware splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)

    # Optional: filter out token-overflow chunks or truncate them
    filtered = []
    for chunk in chunks:
        num_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        if num_tokens <= token_limit:
            filtered.append(chunk.strip())
        else:
            if verbose:
                print(f"⚠️ Chunk exceeded token limit ({num_tokens} > {token_limit}), truncating.")
            token_ids = tokenizer.encode(chunk, add_special_tokens=False)[:token_limit]
            filtered.append(tokenizer.decode(token_ids).strip())

    return filtered
