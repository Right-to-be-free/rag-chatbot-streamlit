import streamlit as st
import requests
import pinecone

# Load credentials securely from Streamlit Secrets
pinecone_api_key = st.secrets["api"]["pinecone_key"]
pinecone_env = st.secrets["api"]["pinecone_env"]
hf_token = st.secrets["api"]["hf_token"]

# Initialize Pinecone
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

# Set Hugging Face Zephyr model endpoint
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# Set request headers for Hugging Face API
headers = {
    "Authorization": f"Bearer {hf_token}"
}

# Function to generate response from LLM via Hugging Face API
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

    result = response.json()
    return result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
