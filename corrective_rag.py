import time
import re
from llm_api import generate_from_api  # Make sure this uses Together API or Zephyr API


def grade_passages(passages, query):
    """Score each passage based on how well it answers the query (Corrective RAG)."""
    print("\n🧪 Grading retrieved passages...")
    graded = []

    for i, doc in enumerate(passages):
        text = (
            doc.get("content", "") or doc.get("chunk_text", "") if isinstance(doc, dict)
            else str(doc)
        )

        prompt = f"""As a legal assistant, assess the relevance of the following passage to the user's question.

Question: {query}

Passage:
{text}

Respond only with a number between 0.0 (not relevant) and 1.0 (very relevant), on a new line.

Score:"""

        try:
            score_text = generate_from_api(prompt)
            print(f"🔎 Raw LLM output [{i+1}]: {score_text}")

            match = re.search(r"\b(\d\.\d+)\b", score_text)
            if not match:
                raise ValueError(f"Could not extract float from: {score_text}")
            score = float(match.group(1))

            if isinstance(doc, dict):
                doc["score"] = score

            graded.append((doc, score))
            print(f"  [{i+1}] Score: {score:.2f}")
        except Exception as e:
            print(f"⚠️ Grading error [{i+1}]: {e}")
            continue

        time.sleep(1.1)  # Safe for Together API (1 QPS limit)

    return sorted(graded, key=lambda x: -x[1])


def reflect_answer(answer, top_docs):
    """Run a Self-RAG check on the generated answer to ensure all claims are supported."""
    print("\n🔍 Running Self-RAG reflection...")

    sources_str = "\n\n".join(
        f"[{i+1}] {doc.get('content', '') if isinstance(doc, dict) else str(doc)}"
        for i, doc in enumerate(top_docs)
    )

    if not sources_str.strip() or len(sources_str.split()) < 20:
        print("⚠️ Warning: Source material is too short or missing. Skipping reflection.")
        return answer

    prompt = f"""You are verifying an AI-generated legal answer using excerpts from a Supreme Court judgment.

Answer:
{answer}

Excerpts:
{sources_str}

Instructions:
- Keep all points that are clearly supported or reasonably inferred from the excerpts.
- Remove or revise only those claims that are clearly speculative or contradict the sources.
- If the answer is fully valid, return it as-is.
- Do not say "not provided" unless there is truly no relevant content.

Revised Answer:"""

    try:
        # Optional: debug logging
        print("\n📝 Reflection Prompt (truncated):\n", prompt[:800], "...\n")

        verified = generate_from_api(prompt)
        print("\n✅ Verified Answer:\n", verified.strip())
        return verified.strip()

    except Exception as e:
        print(f"⚠️ Reflection failed: {e}")
        return answer
