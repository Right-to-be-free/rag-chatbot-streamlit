import nltk
from nltk.tokenize import sent_tokenize
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

def chunk_text_semantic(text: str, model_name: str = "all-MiniLM-L6-v2", overlap: int = 20):
    """
    Splits text into semantically meaningful chunks using sentence boundaries + token-aware limits.

    Args:
        text (str): Full input text.
        model_name (str): Embedding model name.
        overlap (int): Number of tokens to overlap between chunks.

    Returns:
        List[str]: List of text chunks ready for embedding.
    """
    max_tokens = MODEL_TOKEN_LIMITS.get(model_name, 512)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_len = len(sentence_tokens)

        # If sentence alone is too long, split it hard
        if token_len > max_tokens:
            print(f"⚠️ Sentence exceeds model token limit ({token_len} > {max_tokens}), splitting directly.")
            for i in range(0, token_len, max_tokens):
                part_tokens = sentence_tokens[i:i + max_tokens]
                part_text = tokenizer.decode(part_tokens)
                chunks.append(part_text.strip())
            continue

        # If adding sentence exceeds limit, save chunk
        if current_tokens + token_len > max_tokens:
            chunk_text = tokenizer.decode(tokenizer.encode(" ".join(current_chunk), add_special_tokens=False))
            chunks.append(chunk_text.strip())

            # Start new chunk with overlap from end of previous chunk
            if overlap > 0:
                overlap_tokens = tokenizer.encode(" ".join(current_chunk), add_special_tokens=False)[-overlap:]
                current_chunk = [tokenizer.decode(overlap_tokens)]
                current_tokens = len(overlap_tokens)
            else:
                current_chunk = []
                current_tokens = 0

        # Append sentence to current chunk
        current_chunk.append(sentence)
        current_tokens += token_len

    # Final chunk
    if current_chunk:
        chunk_text = tokenizer.decode(tokenizer.encode(" ".join(current_chunk), add_special_tokens=False))
        chunks.append(chunk_text.strip())

    return chunks
