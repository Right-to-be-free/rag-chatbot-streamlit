import os
import requests

class TogetherLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", max_tokens=512, temperature=0.7):
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("❌ TOGETHER_API_KEY not set in environment variables.")
        
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": None
        }

        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Together API error {response.status_code}: {response.text}")
        
        data = response.json()
        return data["choices"][0]["text"].strip()
