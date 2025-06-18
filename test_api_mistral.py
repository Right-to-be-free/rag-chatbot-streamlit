from llm_api import generate_from_api

prompt = """Answer the following question based on the provided context.

Context:
Kafka is a distributed streaming platform used for building real-time data pipelines and streaming apps.
Producers write data to Kafka topics and consumers read from them, allowing decoupling of services.
Kafka guarantees message durability and high throughput, making it suitable for large-scale systems.

Question: How does Kafka help in real-time data processing?

Answer:"""

response = generate_from_api(prompt)
print("\n📢 Response:\n", response)
