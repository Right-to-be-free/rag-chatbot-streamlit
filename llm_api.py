import os
import requests
from dotenv import load_dotenv

load_dotenv()  # ✅ Must be called before getenv

HF_API_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"



headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def generate_from_api(prompt: str, max_tokens=256):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.json()[0]['generated_text']
