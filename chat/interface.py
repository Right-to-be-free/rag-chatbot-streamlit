from llm_together import TogetherLLM
from llm_openrouter import OpenRouterLLM
from llm_cohere import CohereLLM  # only if using

class LLMInterface:
    def __init__(self, llm_type="together", model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.history = []

        if llm_type == "together":
            self.llm = TogetherLLM(model_name=model_name)
        elif llm_type == "openrouter":
            self.llm = OpenRouterLLM(model_name=model_name)
        elif llm_type == "cohere":
            self.llm = CohereLLM(model_name=model_name)  # optional
        else:
            raise ValueError(f"❌ Unsupported LLM type: {llm_type}")

    def ask(self, question, context=""):
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}
Answer:"""
        response = self.llm.generate(prompt)
        self.history.append({"question": question, "response": response})
        return response
