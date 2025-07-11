def build_rag_prompt(query: str, retrieved_chunks: list, max_tokens: int = 1500) -> str:
    """
    Build a prompt for RAG by combining retrieved document chunks with a query.
    
    Args:
        query (str): User's question or input prompt.
        retrieved_chunks (list): List of dicts with text from vector search.
        max_tokens (int): Max token budget for context.

    Returns:
        str: Final prompt with injected context and query.
    """
    context_blocks = []
    total_length = 0

    for chunk in retrieved_chunks:
        text = chunk.get("text") or chunk.get("chunk") or chunk.get("file_path", "")
        if not text:
            continue
        if total_length + len(text) > max_tokens:
            break
        context_blocks.append(text)
        total_length += len(text)

    context_text = "\n\n".join(context_blocks)

    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
