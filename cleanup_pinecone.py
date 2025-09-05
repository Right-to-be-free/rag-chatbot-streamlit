import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "all-minilm-l6-v2-384"  # Update if your index name is different
NAMESPACE = "__default__"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
index = pc.Index(INDEX_NAME)

# Optional: show total vectors before
stats = index.describe_index_stats()
print(f"📊 Vectors before deletion: {stats.get('total_vector_count', 'N/A')}")

# Step 1: Get all IDs (up to 1000)
print(f"🔍 Fetching vector IDs in namespace '{NAMESPACE}'...")
query_result = index.query(
    vector=[0.0] * 384,  # dummy vector
    top_k=1000,
    include_values=False,
    namespace=NAMESPACE
)
ids = [match['id'] for match in query_result.get("matches", [])]

# Step 2: Delete
if ids:
    print(f"🗑️ Deleting {len(ids)} vectors...")
    index.delete(ids=ids, namespace=NAMESPACE)
    print("✅ Deletion complete.")
else:
    print("✅ No vectors to delete.")
