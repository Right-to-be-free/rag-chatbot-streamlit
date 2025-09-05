import os
import cohere

class CohereLLM:
    def __init__(self, model_name="command-r-plus"):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("❌ COHERE_API_KEY is not set in the environment.")
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens=300) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.generations[0].text.strip()
