import os
import time
import requests

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
API_URL = "https://api.together.xyz/v1/completions"

def generate_from_api(prompt: str, max_tokens=1024, temperature=0.3, retries=3, backoff=2):
    if not TOGETHER_API_KEY:
        raise ValueError("Set TOGETHER_API_KEY in your environment.")

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|endoftext|>"]
    }

    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 429:
            wait = backoff * (attempt + 1)
            print(f"⚠️ Rate limit hit (429). Retrying in {wait} seconds...")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        return response.json()["choices"][0]["text"].strip()

    # If all retries exhausted
    raise Exception("❌ Failed after retrying due to persistent rate limits.")
