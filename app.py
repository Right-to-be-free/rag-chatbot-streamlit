import streamlit as st
from document_manager import DocumentManager
from chat.interface import LLMInterface

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

# Vector DB Selection
db_type = st.sidebar.selectbox("Select Vector DB", ["pinecone", "faiss", "chroma", "mongodb", "qdrant"])

# Embedding Model Selection
model_name = st.sidebar.selectbox("Select Embedding Model", [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "distilbert-base-nli-stsb-mean-tokens",
    "bert-base-nli-mean-tokens",
    "roberta-base-nli-mean-tokens"
])

# LLM Service Selection with default models
llm_map = {
    "Together AI": ("together", "mistralai/Mistral-7B-Instruct-v0.2"),
    "OpenRouter": ("openrouter", "openai/gpt-3.5-turbo"),
    "Cohere": ("cohere", "command-r-plus")
}

llm_choice = st.sidebar.selectbox("Select LLM Service", list(llm_map.keys()))
llm_type, default_llm_model = llm_map[llm_choice]

llm_model = st.sidebar.text_input("LLM Model Name", value=default_llm_model)

# Initialize Document Manager and LLM
doc_manager = DocumentManager(db_type=db_type, model_name=model_name)
llm_interface = LLMInterface(llm_type=llm_type, model_name=llm_model)

# Main Title
st.title("🧠 RAG-powered QA Chatbot")

# User Input
user_query = st.text_input("Ask a question about your documents:")
submit_button = st.button("Submit")

# Main Logic
if submit_button and user_query:
    with st.spinner("Retrieving relevant context..."):
        results = doc_manager.query(user_query)
        if not results:
            st.warning("No relevant documents found.")
        else:
            context = "\n\n".join([
                f"[{i+1}] {doc.get('metadata', {}).get('chunk_text', '') if isinstance(doc, dict) else str(doc)}"
                for i, doc in enumerate(results)
            ])

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

            with st.spinner("Generating response..."):
                answer = llm_interface.ask(user_query, context=context)
                st.success("💬 Answer:")
                st.write(answer)
