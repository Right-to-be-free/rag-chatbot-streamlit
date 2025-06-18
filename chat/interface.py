from llm_api import generate_from_api

class LLMInterface:
    def __init__(self):
        self.history = []

    def ask(self, question: str, context: str = "") -> str:
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}
Answer:"""

        response = generate_from_api(prompt)
        self.history.append({"question": question, "response": response})
        return response
